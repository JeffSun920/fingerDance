


import cv2
import numpy as np
import mediapipe as mp
import pygame

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 初始化Pygame
pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Finger Dance')
clock = pygame.time.Clock()

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 初始化绘图面板
screen.fill(WHITE)

# 打开摄像头
cap = cv2.VideoCapture(0)
cap_width, cap_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

def erase(img, center, size):
    top_left = (center[0] - size // 2, center[1] - size // 2)
    bottom_right = (center[0] + size // 2, center[1] + size // 2)
    pygame.draw.rect(img, WHITE, (*top_left, size, size))

def draw_circle(img, center, color):
    pygame.draw.circle(img, color, center, 20, 2)

def draw_line(img, start, end, color):
    pygame.draw.line(img, color, start, end, 2)

def main():
    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8) as hands:

        drawing = False
        erasing = False
        circle_drawing = False
        prev_pos = None

        while True:
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    return

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = [(int(point.x * screen_width), int(point.y * screen_height)) for point in
                                 hand_landmarks.landmark]

                    # 检查不同手势并确保功能独立运行
                    if all(landmarks[i][1] < landmarks[i - 2][1] for i in range(8, 21, 4)):
                        # 手掌完全打开（橡皮擦功能）
                        erasing = True
                        drawing = False
                        circle_drawing = False
                        erase(screen, landmarks[9], 50)
                    elif landmarks[8][1] < landmarks[7][1] < landmarks[6][1] and \
                            all(landmarks[i][1] > landmarks[i - 2][1] for i in range(12, 21, 4)):
                        # 单食指指向（画线功能）
                        drawing = True
                        erasing = False
                        circle_drawing = False
                        if prev_pos is not None:
                            draw_line(screen, prev_pos, landmarks[8], BLACK)
                        prev_pos = landmarks[8]
                    else:
                        drawing = False
                        prev_pos = None

            # 显示摄像头图像
            cv2.imshow('Camera', image)

            pygame.display.flip()
            clock.tick(30)

            if cv2.waitKey(5) & 0xFF == 27:
                break

if __name__ == "__main__":
    main()
