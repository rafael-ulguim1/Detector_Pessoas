import cv2

def detectar_pessoas():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture('./src/ia_m_uv/videos/pessoas3.mp4')

    if not cap.isOpened():
        print("Erro ao abrir a câmera/vídeo")
        return

    total_pessoas_detectadas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)

        total_pessoas_detectadas += len(rects)  # soma quantas pessoas foram detectadas neste frame

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar o número de pessoas detectadas no frame
        cv2.putText(frame, f"Pessoas detectadas: {len(rects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Detecção de Pessoas', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total de pessoas detectadas no vídeo (soma por frame): {total_pessoas_detectadas}")

if __name__ == "__main__":
    detectar_pessoas()
