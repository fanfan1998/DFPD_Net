import numpy as np
import cv2

def apply_mask_to_image(forgery_mask, forgery_image, intensity=1.5):
    if forgery_image.ndim == 3 and forgery_image.shape[2] == 1:
        forgery_image = forgery_image[:, :, 0]
    forgery_mask = forgery_mask.astype(bool)
    masked_image = forgery_image.copy()
    masked_image[forgery_mask] = (forgery_image[forgery_mask] * intensity).clip(0, 255).astype(np.uint8)
    return masked_image

class GradCAM(object):
    def __init__(self, net, layer_name, pred_key):
        self.net = net
        self.layer_name = layer_name
        self.pred_key = pred_key
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def __call__(self, inputs, index, img_size):
        self.net.zero_grad()
        output_dict = self.net(inputs)
        output = output_dict[self.pred_key]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature[0].cpu().data.numpy()
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)
        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, img_size)
        return cam, index
