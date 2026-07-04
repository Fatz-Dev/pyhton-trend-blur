import cv2
import mediapipe as mp

# ---------------------------
# Inisialisasi MediaPipe Hands
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def finger_up(tip, pip, landmarks):
    """Cek apakah ujung jari lebih tinggi (y lebih kecil) dari sendi pip-nya."""
    return landmarks[tip].y < landmarks[pip].y


def thumb_up(landmarks, handedness_label="Right"):
    """
    Cek apakah ibu jari terentang (bukan ke arah tangan / menempel).
    Ibu jari bergerak lebih horizontal, jadi kita bandingkan sumbu x
    antara tip (4) dan sendi ip (3), relatif ke arah tangan.
    """
    tip_x = landmarks[4].x
    ip_x = landmarks[3].x
    mcp_x = landmarks[2].x

    # Untuk tangan kanan (hasil flip kamera), ibu jari terentang ke kiri (x tip < x mcp)
    if handedness_label == "Right":
        return tip_x < mcp_x and tip_x < ip_x
    else:
        return tip_x > mcp_x and tip_x > ip_x


def is_peace(landmarks, handedness_label="Right"):
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)
    thumb_folded = not thumb_up(landmarks, handedness_label)

    return (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
        and thumb_folded
    )


def get_blur_kernel(frame_width, base_ratio=10, min_size=31):
    """Hitung ukuran kernel blur yang proporsional dengan lebar frame (harus ganjil)."""
    k = max(min_size, frame_width // base_ratio)
    if k % 2 == 0:
        k += 1
    return (k, k)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera tidak bisa diakses. Cek koneksi/permission kamera.")
        return

    # Set resolusi lebih ringan agar deteksi lebih responsif (opsional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Gagal membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_result = hands.process(rgb)
            peace_detected = False

            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hand_result.multi_hand_landmarks,
                    hand_result.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" atau "Right"
                    if is_peace(hand_landmarks.landmark, label):
                        peace_detected = True
                        break

            # Efek blur jika gestur peace terdeteksi
            if peace_detected:
                kernel = get_blur_kernel(frame.shape[1])
                frame = cv2.GaussianBlur(frame, kernel, 0)
                cv2.putText(
                    frame,
                    "Blur!",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Peace Blur", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
                break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()