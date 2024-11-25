import torch


def get_average_image(net, opts):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)['image'][0]
    avg_image = avg_image.to('cuda').float().detach()
    if opts.dataset_type == "cars_encode":
        avg_image = avg_image[:, 32:224, :]
    return avg_image


def run_on_batch(inputs, net, opts, avg_image, return_features=False, return_latents=False):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    results_features = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        res = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=return_latents,
                                    return_features=return_features,
                                    resize=opts.resize_outputs)

        y_hat = res['image']

        if opts.dataset_type == "cars_encode":
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            if return_latents: results_latent[idx].append(res['latent'][idx])
            if return_features: results_features[idx].append([f[idx]for f in res['features']])

        # resize input to 256 before feeding into next iteration
        if opts.dataset_type == "cars_encode":
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    return dict(images=results_batch, latent=results_latent, features=results_features)
