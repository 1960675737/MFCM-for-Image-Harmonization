import cv2
import os.path as osp
import matplotlib.pyplot as plt

if __name__=="__main__":
    # image_dir = "./images"
    # image_name = 'd17664-20120114-145839.jpg'
    # image_path = osp.join(image_dir, image_name)
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_hsv_0_0 = img_hsv[:, :, 0]
    # img_hsv_0_1 = img_hsv[:, :, 1]
    # img_hsv_0_2 = img_hsv[:, :, 2]
    #
    # image_name_1 = 'd17664-20120114-145839_1_4.jpg'
    # image_path_1 = osp.join(image_dir, image_name_1)
    # img_1 = cv2.imread(image_path_1)
    # img_1 = cv2.resize(img_1, (256, 256), cv2.INTER_LINEAR)
    # img_hsv_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
    # img_hsv_1_0 = img_hsv_1[:, :, 0]
    # img_hsv_1_1 = img_hsv_1[:, :, 1]
    # img_hsv_1_2 = img_hsv_1[:, :, 2]
    #
    # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    # cv2.namedWindow('img', 2)
    # cv2.imshow('img', img)
    # cv2.namedWindow('img_hsv', 2)
    # cv2.imshow('img_hsv', img_hsv)
    # cv2.namedWindow('img_h', 2)
    # cv2.imshow('img_h', img_hsv_0_0)
    # cv2.namedWindow('img_s', 2)
    # cv2.imshow('img_s', img_hsv_0_1)
    # cv2.namedWindow('img_v', 2)
    # cv2.imshow('img_v', img_hsv_0_2)
    #
    # cv2.namedWindow('img_1', 2)
    # cv2.imshow('img_1', img_1)
    # cv2.namedWindow('img_hsv_1', 2)
    # cv2.imshow('img_hsv_1', img_hsv_1)
    # cv2.namedWindow('img_h_1', 2)
    # cv2.imshow('img_h_1', img_hsv_1_0)
    # cv2.namedWindow('img_s_1', 2)
    # cv2.imshow('img_s_1', img_hsv_1_1)
    # cv2.namedWindow('img_v_1', 2)
    # cv2.imshow('img_v_1', img_hsv_1_2)
    # cv2.waitKey(0)

    image_dir = "./images"
    image_name = 'd90000014-10.jpg'
    image_path = osp.join(image_dir, image_name)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_B = img[:, :, 0]
    img_G = img[:, :, 1]
    img_R = img[:, :, 2]

    image_name_1 = 'd90000014-10_1_9.jpg'
    image_path_1 = osp.join(image_dir, image_name_1)
    img_1 = cv2.imread(image_path_1)
    img_1 = cv2.resize(img_1, (256, 256), cv2.INTER_LINEAR)
    # img_hsv_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
    img_1_B = img_1[:, :, 0]
    img_1_G = img_1[:, :, 1]
    img_1_R = img_1[:, :, 2]

    cv2.namedWindow('img', 2)
    cv2.imshow('img', img)
    cv2.namedWindow('img_B', 2)
    cv2.imshow('img_B', img_B)
    cv2.namedWindow('img_G', 2)
    cv2.imshow('img_G', img_G)
    cv2.namedWindow('img_R', 2)
    cv2.imshow('img_R', img_R)

    cv2.namedWindow('img_1', 2)
    cv2.imshow('img_1', img_1)
    cv2.namedWindow('img_1_B', 2)
    cv2.imshow('img_1_B', img_1_B)
    cv2.namedWindow('img_1_G', 2)
    cv2.imshow('img_1_G', img_1_G)
    cv2.namedWindow('img_1_R', 2)
    cv2.imshow('img_1_R', img_1_R)
    cv2.waitKey(0)


    # cv.destroyAllWindows()
    # plt.figure()
    # plt.subplot(231)
    # plt.imshow(img)
    # plt.subplot(232)
    # plt.imshow(img_hsv)

    # plt.subplot(234)
    # plt.imshow(img_hsv_0)
    # plt.subplot(235)
    # plt.imshow(img_hsv_1)
    # plt.subplot(236)
    # plt.imshow(img_hsv_2)
    # plt.show()
    # cv2.imshow('img', img_hsv)
    # cv2.waitKey(0)
    # cv.destroyAllWindows()