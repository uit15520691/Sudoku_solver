# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:57:57 2018

@author: qq
"""

import cv2
import numpy as np
import joblib
import sudoku_solver
import os

PI = 3.1415926535897932384626433832795
font = cv2.FONT_HERSHEY_DUPLEX
clf = joblib.load('classifier.pkl')
folder = "sudoku"

for file in os.listdir(folder):
    filepath = os.path.join(folder, file)
    print filepath
    original = cv2.imread(filepath)
    
    h, w = original.shape[:2]
    scale=float(h)/w
    print scale
    if (scale>1):
        original = cv2.resize(original, (h,w), 1.0/scale, 1)
    else:
        original = cv2.resize(original, (h,w), 1, 1.0/scale)
 #   original = cv2.resize(original, (600, 600))
    frame = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0., 1, 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    sudoku1 = cv2.dilate(frame, kernel)
    sudoku1 = cv2.blur(frame, (3,3))
    edges = cv2.Canny(sudoku1, 70, 200)
    lines = cv2.HoughLines(edges, 2, PI /180, 300)
    if (lines is not None):
        lines = lines[0]
        lines = sorted(lines, key=lambda line:line[0])
        diff_ngang = 0
        diff_doc = 0
        lines_1=[]
        Points=[]
        for rho,theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            if (b>0.5):
                if(rho-diff_ngang>8):
                    diff_ngang=rho
                    lines_1.append([rho,theta, 0])
                    
            else:
                if(rho-diff_doc>8):
                    diff_doc=rho
                    lines_1.append([rho,theta, 1])
                    

        for i in range(len(lines_1)):
            if(lines_1[i][2] == 0):
                for j in range(len(lines_1)):
                    if (lines_1[j][2]==1):
                        theta1=lines_1[i][1]
                        theta2=lines_1[j][1]
                        p1=lines_1[i][0]
                        p2=lines_1[j][0]
                        xy = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                        p = np.array([p1,p2])
                        res = np.linalg.solve(xy, p)
                        Points.append(res)
        print 'number of Points',len(Points)                
        solved=0
        if(len(Points)==100):
            result = []
            board = []
            
            sudoku1 = cv2.adaptiveThreshold(sudoku1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 101, 1)
            
            count =0
            for i in range(0,9):
                for j in range(0,9):
                    y1=int(Points[j+i*10][1]+5)
                    y2=int(Points[j+i*10+11][1]-5)
                    x1=int(Points[j+i*10][0]+5)
                    x2=int(Points[j+i*10+11][0]-5)
                    X = sudoku1[y1:y2,x1:x2]
                    
                    if(X.size!=0):
                        X = cv2.resize(X, (36,36))
                        num = clf.predict(np.reshape(X, (1,-1)))
                        result.append(num)
                        board.append(num)
            
            result = np.reshape(result, (9,9))
            board = np.reshape(board, (9,9))
            print 'dang giai sudoku'
            solved=sudoku_solver.solveSudoku(result)
            if(solved):
                frame = original.copy()
                for i in range(0,9):
                    for j in range(0,9):
                        if(board[i][j]==0):
                           cv2.putText(frame,str(result[i][j]),
                                       (int(Points[j+i*10+10][0]+15),int(Points[j+i*10+10][1]-10)),
                                       font,1,(78 , 69, 255),2)
    
    original = cv2.resize(original, (300, 300)) 
    frame = cv2.resize(frame, (300, 300)) 
    sudoku1 = cv2.cvtColor(sudoku1,cv2.COLOR_GRAY2RGB)
    sudoku1 = cv2.resize(sudoku1, (300, 300)) 
    if (solved==1):
        stack = np.hstack((original,sudoku1, frame))
    else:
        print "thuat toan khong giai duoc\n--------"
        stack = np.hstack((original,sudoku1))
    cv2.imshow('sudoku - solved',stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()