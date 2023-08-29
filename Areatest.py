
'''
-. select file
-. get frame by opencv

'''
import cv2

def getCircleContour(imgcopy):
    def is_center(contour):
        center_diff = 250
        M = cv2.moments(contour)
        M0 = 1 if M["m00"] == 0 else M["m00"]
        center = (int(M["m10"] / M0), int(M["m01"] / M0))
        img_center = (img_gray.shape[1] // 2, img_gray.shape[0] // 2)
        if abs(img_center[0] - center[0]) > center_diff or abs(img_center[1] - center[1]) > center_diff:
            return False
        return True

    def is_circle(contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        _, (x, y), _ = cv2.fitEllipse(contour)
        ellipse_ratio = x / y
        if aspect_ratio > 0.7 and aspect_ratio < 1.3 and ellipse_ratio > 0.7 and ellipse_ratio < 1.2:
            return True
        else:
            return False

    conlist = []
    img_gray = cv2.cvtColor(imgcopy, cv2.COLOR_BGR2GRAY)
    img_gray = remove_noise(img_gray)

    temp_thres = np.mean(
        img_gray[img_gray.shape[0] // 2][img_gray.shape[1] // 2 - 30:img_gray.shape[1] // 2 + 30]) * 0.9

    ret, thr = cv2.threshold(img_gray, int(temp_thres), 255, cv2.THRESH_BINARY)

    contour1, dst = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contour1:
        if 2000 > contour.size > 500 and is_center(contour):
            if is_circle(contour):
                conlist.append(contour)
    # if cnt == 200:
    #    cv2.drawContours(imgcopy, [conlist[0]], -1, (0,0,255), -1)
    #    cv2.imwrite(f"{cnt}_{random.randint(1,100)}.png", imgcopy)
    return conlist, thr


# ---------------------- 위는 참고 ----------------------


# def createLog():
#     import xlsxwriter
#     wb = xlsxwriter.Workbook("./result_log.xlsx")
#     # index_ws = index_wb.add_worksheet("Index")
#     ws = list()
#     header = ['VisionName', 'PArea', 'PLen', 'Pxline', 'Pyline', 'Px', 'Py', 'Penclosing_x', 'Penclosing_y',
#               # 'P_PCA_angle', 'P_PCA_x', 'P_PCA_y', 'P_p1_x', 'P_p1_y','P_p2_x', 'P_p2_y',
#               'CArea', 'Clen', 'CxLine', 'CyLine', 'Cx', 'Cy', 'Cenclosing_x',
#               'Cenclosing_y']  # , 'C_PCA_angle', 'C_PCA_x', 'C_PCA_y', 'C_p1_x', 'C_p1_y', 'C_p2_x', 'C_p2_y']
#     header = ['Current_time', 'Score', 'Process_time']
#     ws = wb.add_worksheet(typeName)
#     ws.write_row('A1', header,
#                  wb.add_format({'bold': 1, 'font_size': 10, 'bg_color': 'silver', 'align': 'center', 'border': 1}))
#     return wb, ws

def select_file(init_dir='./', file_type=(("All files", "*.*"),)):
    import tkinter
    import os
    from tkinter import filedialog
    '''
    .. Note:: 1개의 파일 선택
    :return: filename (str) : 선택된 Filename (경로 포함)
    '''
    tk_root = tkinter.Tk()
    tk_root.withdraw()
    filename = os.path.abspath(filedialog.askopenfilename(parent=tk_root,
                                                          initialdir=init_dir,
                                                          title='Please select a file',
                                                          filetypes=file_type))
    # print("Selected file: {}".format(filename))
    # logging.info('Selected file: {}'.format(filename))
    return filename
def _remove_noise(img, kernel_size=3):
    result = cv2.GaussianBlur(img, (kernel_size,kernel_size),0 )
    return result
def _find_polygon_vertices(contour):
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx
def _get_score(frame):
    # import cv2

    import matplotlib.pyplot as plt

def _convert2binary(img):
    import numpy as np
    # to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get Threshold
    thres = np.mean(img_gray)/2

    # get binary
    ret, img_thres = cv2.threshold(img_gray, int(thres), 255, cv2.THRESH_BINARY)

    return img_thres


def _find_contour(img):
    def _find_center(contour):
        M = cv2.moments(contour)
        if int(M['m00']) == 0: # Case of dot
            return 0,0
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        return center_x, center_y

    def _p2p_distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**1/2
    def _close2center(contour):
        tmp_x, tmp_y = _find_center(contour)
        # cal distance from img center
        if _p2p_distance((tmp_x, tmp_y), (img_cx, img_cy) ) < _p2p_distance((cx, cy), (img_cx, img_cy)):
            return True, (tmp_x, tmp_y)
        return False, (0,0)
    # get contour
    contours, dst = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, 0, 0
    # size filtering
    container = []
    for con in contours:
        if con.size > 100:
            container.append(con)

    # get image center
    H, W = img.shape
    img_cy, img_cx = H/2, W/2

    # select contour close to center
    for idx, contour in enumerate(container):
        if idx == 0:
            c = contour
            cx, cy = _find_center(c)
        else:
            ret, result = _close2center(contour)
            if ret:
                c = contour
                cx, cy = result
    return c, cx, cy


def hexagon_test(device):
    # import cv2
    import time

    cap = cv2.VideoCapture(device)

    while True:
        # check start time
        start_time = time.time()
        # get frame
        ret, frame = cap.read()

        if ret: # if Success
            # img for save
            output_img = frame.copy()
            # convert to binary
            img_binary = _convert2binary(img=frame)

            # get contours
            contour, cx, cy = _find_contour(img=img_binary)
            # cv2.drawContours(output_img, contour, -1, (0, 255, 0), 3)
            # cv2.circle(output_img, (cx, cy), 5, (0, 255, 0), -1)
            # cv2.imwrite("1_contour.jpg", output_img)

            # get vertices
            vertices = _find_polygon_vertices(contour)
            # 검출된 꼭짓점을 원본 이미지에 그리기
            for vertex in vertices:
                x, y = vertex[0]
                cv2.circle(output_img, (x, y), 5, (255, 0, 0), -1)
            cv2.imwrite("2_vertices.jpg", output_img)

            # get score
            _get_score(frame)
        else:   # if Failed
            raise Exception("Failed to read Frame !")

        # check end time
        end_time = time.time()

        # check process time
        process_time = end_time - start_time
        print(f"Process time : {process_time}")



if __name__ == "__main__":
    # file 선택
    f = select_file(init_dir='./1-1')

    # hexagon 테스트
    hexagon_test(f)


    print("Finished")