from torchvision import transforms as T

preprocessing_func = T.Compose(
    [
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def preprocess(img):
    return preprocessing_func(img)
