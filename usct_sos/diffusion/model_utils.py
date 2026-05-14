from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random, vmap
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


def _unpack_batch(batch):
    """Support both old `(coords, x, y)` batches and new `((coords, x, y), rf, probe)` batches."""
    if isinstance(batch, (tuple, list)) and len(batch) == 3 and isinstance(batch[0], (tuple, list)):
        coords, x, y = batch[0]
        rf = batch[1]
        probe = batch[2]
        return coords, x, y, rf, probe

    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        coords, x, y = batch
        return coords, x, y, None, None

    raise ValueError(f"Unsupported batch structure: {type(batch)} / len={len(batch) if hasattr(batch, '__len__') else 'NA'}")


def _decode_points(decoder, decoder_params, z, coords):
    coords = jnp.squeeze(coords)
    if coords.ndim == 1:
        coords = coords[None, :]

    def _decode_one(coord):
        pred = decoder.apply(decoder_params, z, coord)
        return jnp.squeeze(pred)

    preds = vmap(_decode_one, in_axes=0, out_axes=1)(coords)
    if preds.ndim == 2:
        preds = preds[..., None]
    return preds


def _loss_fn(
    encoder,
    decoder,
    params,
    batch,
    rng=None,
    use_pde=False,
    drop_prob=0.0,
    training=False,
    force_cond=False,
):
    encoder_params, decoder_params = params
    coords, x, y, rf, probe = _unpack_batch(batch)

    if rf is not None and probe is not None:
        z = encoder.apply(
            encoder_params,
            x,
            rf=rf,
            probe=probe,
            rng=rng,
            drop_prob=drop_prob,
            training=training,
            force_cond=force_cond,
        )
    else:
        z = encoder.apply(encoder_params, x)

    y = jnp.squeeze(y)
    u_pred = _decode_points(decoder, decoder_params, z, coords)
    u_pred = jnp.squeeze(u_pred)

    #loss_data = jnp.mean((y - u_pred) ** 2)
    loss_data = jnp.mean(jnp.abs(y - u_pred)) #L1 loss
    loss_res = jnp.asarray(0.0, dtype=loss_data.dtype)
    loss = loss_data if not use_pde else loss_data
    return loss, (loss_data, loss_res)


def create_train_step(
    encoder,
    decoder,
    mesh,
    use_pde=False,
    drop_prob=0.2,
    training=True,
    force_cond=False,
    return_grad_norms=False,
):
    if return_grad_norms:
        @jax.jit
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(), P("batch"), P()),
            out_specs=(P(), P(), P(), P(), P()),
            check_rep=False,
        )
        def train_step(state, batch, rng=None):
            grad_fn = jax.value_and_grad(
                partial(
                    _loss_fn,
                    encoder,
                    decoder,
                    use_pde=use_pde,
                    drop_prob=drop_prob,
                    training=training,
                    force_cond=force_cond,
                ),
                has_aux=True,
            )
            (loss, aux), grads = grad_fn(state.params, batch, rng)
            loss_data, loss_res = aux

            grads = lax.pmean(grads, "batch")

            loss = lax.pmean(loss, "batch")
            loss_data = lax.pmean(loss_data, "batch")
            loss_res = lax.pmean(loss_res, "batch")

            state = state.apply_gradients(grads=grads)
            return state, loss, loss_data, loss_res, grads

        return train_step

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch"), P()),
        out_specs=(P(), P(), P(), P()),
        check_rep=False,
    )
    def train_step(state, batch, rng=None):
        grad_fn = jax.value_and_grad(
            partial(
                _loss_fn,
                encoder,
                decoder,
                use_pde=use_pde,
                drop_prob=drop_prob,
                training=training,
                force_cond=force_cond,
            ),
            has_aux=True,
        )
        (loss, aux), grads = grad_fn(state.params, batch, rng)
        loss_data, loss_res = aux

        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        loss_data = lax.pmean(loss_data, "batch")
        loss_res = lax.pmean(loss_res, "batch")

        state = state.apply_gradients(grads=grads)
        return state, loss, loss_data, loss_res

    return train_step


def create_encoder_step(encoder, mesh, training=False, drop_prob=0.0, force_cond=False):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch"), P()),
        out_specs=P("batch"),
        check_rep=False,
    )
    def encoder_step(encoder_params, batch, rng=None):
        coords = x = y = rf = probe = None
        if isinstance(batch, (tuple, list)) and len(batch) == 3 and not isinstance(batch[0], (tuple, list)):
            # Compact form for diffusion training: (x, rf, probe)
            x, rf, probe = batch
        else:
            coords, x, y, rf, probe = _unpack_batch(batch)
        if rf is not None and probe is not None:
            z = encoder.apply(
                encoder_params,
                x,
                rf=rf,
                probe=probe,
                rng=rng,
                drop_prob=drop_prob,
                training=training,
                force_cond=force_cond,
            )
        else:
            z = encoder.apply(encoder_params, x)
        return z

    return encoder_step


def create_decoder_step(decoder, mesh):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch"), P("batch")),
        out_specs=P("batch"),
        check_rep=False,
    )
    def decoder_step(decoder_params, z, coords):
        return _decode_points(decoder, decoder_params, z, coords)

    return decoder_step


def create_eval_step(encoder, decoder, mesh, use_pde=False):
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch"), P()),
        out_specs=(P("batch"), P(), P(), P()),
        check_rep=False,
    )
    def eval_step(state, batch, rng=None):
        loss, (loss_data, loss_res) = _loss_fn(
            encoder,
            decoder,
            state.params,
            batch,
            rng=rng,
            use_pde=use_pde,
            drop_prob=0.0,
            training=False,
            force_cond=False,
        )
        coords, x, y, rf, probe = _unpack_batch(batch)
        if rf is not None and probe is not None:
            z = encoder.apply(state.params[0], x, rf=rf, probe=probe, rng=rng, training=False, force_cond=False)
        else:
            z = encoder.apply(state.params[0], x)
        pred = _decode_points(decoder, state.params[1], z, coords)
        return pred, loss, loss_data, loss_res

    return eval_step

