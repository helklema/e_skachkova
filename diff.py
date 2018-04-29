import sys
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def diff_remove_bg(img0, img, img1):
    d1 = diff(img0, img)
    d2 = diff(img, img1)
    return cv2.bitwise_and(d1, d2)


def args_parser():
    args = sys.argv
    if len(args) < 3:
        print "Not enough files"
        sys.exit(1)
    return args[1], args[2]


if __name__ == "__main__":
    file1, file2 = args_parser()
    x1 = cv2.imread(file1)
    x2 = cv2.imread(file2)

    x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
    x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)

    absdiff = cv2.absdiff(x1, x2)
    cv2.imwrite("absdiff.png", absdiff)

    diff = cv2.subtract(x1, x2)
    result = not np.any(diff)

    m = mse(x1, x2)
    s = ssim(x1, x2)

    print "mse: %s, ssim: %s" % (m, s)

    if result:
        print "The images are the same"
    else:
        cv2.imwrite("diff.png", diff)
        print "The images are different"
