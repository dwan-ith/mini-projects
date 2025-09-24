import face_recognition
import cv2
import os

def load_known_faces(known_faces_dir="known_faces"):
    known_face_encodings = []
    known_face_names = []
    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(known_faces_dir, file_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file_name)[0])
    return known_face_encodings, known_face_names

def recognize_faces_in_image(image_path, known_face_encodings, known_face_names):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image_cv, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image_cv, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Face Recognition", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_faces_dir = "known_faces"
    test_image = "test.jpg"
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    recognize_faces_in_image(test_image, known_face_encodings, known_face_names)

