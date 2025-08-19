import frappe
import face_recognition
from PIL import Image, ExifTags
from frappe.utils import now_datetime

@frappe.whitelist()
def face_checkin_uploaded(uploaded_file_url):
    """
    Compare uploaded snapshot with all active employees' images
    and mark Employee Checkin if a match is found.
    Handles auto-rotation if face is not detected.
    uploaded_file_url: /files/... or private file URL from ERPNext
    """
    try:
        # Get full filesystem path of uploaded image
        uploaded_img_path = get_full_path(uploaded_file_url)
        if not uploaded_img_path:
            return {"status": "error", "msg": "Uploaded file not found"}

        # Load uploaded snapshot with auto-rotation
        uploaded_encoding = load_image_with_rotation(uploaded_img_path)
        if uploaded_encoding is None:
            return {"status": "error", "msg": "No face detected in uploaded image."}

        # Fetch all active employees with images
        employees = frappe.get_all(
            "Employee",
            filters={"status": "Active"},
            fields=["name", "image"]
        )

        for emp in employees:
            if not emp.image:
                continue

            emp_img_path = get_full_path(emp.image)
            if not emp_img_path:
                continue

            try:
                # Load employee image with auto-rotation
                emp_encoding = load_image_with_rotation(emp_img_path)
                if emp_encoding is None:
                    continue

                # Compare faces
                results = face_recognition.compare_faces([emp_encoding], uploaded_encoding, tolerance=0.54)
                if len(results) > 0 and results[0]:
                    # Match found → create checkin
                    checkin_doc = frappe.get_doc({
                        "doctype": "Employee Checkin",
                        "employee": emp.name,
                        "log_type": "IN",
                        "time": now_datetime()
                    })
                    checkin_doc.insert(ignore_permissions=True)
                    frappe.db.commit()
                    return {"status": "success", "employee": emp.name, "msg": "Check-in marked successfully"}

            except Exception as e:
                frappe.log_error(frappe.get_traceback(), f"Error processing image for {emp.name}")
                continue

        return {"status": "error", "msg": "No matching employee found"}

    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Face Checkin Error")
        return {"status": "error", "msg": str(e)}


# -----------------------
# Helper Functions
# -----------------------

def get_full_path(file_url):
    """
    Convert ERPNext file URL (/files/... or private) to full filesystem path
    """
    try:
        if file_url.startswith("/files/") or file_url.startswith("/private/files/"):
            file_doc = frappe.get_doc("File", {"file_url": file_url})
            return file_doc.get_full_path()
        elif file_url.startswith("/public/"):
            return frappe.get_site_path() + file_url
        else:
            return frappe.get_site_path() + "/public" + file_url
    except Exception as e:
        frappe.log_error(str(e), "Get Full Path Error")
        return None


def load_image_with_rotation(img_path):
    """
    Try to load and encode an image.
    If no face is detected, rotate in all 90° increments.
    Returns face encoding or None if no face found.
    """
    def get_encoding(path):
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        return encodings[0] if encodings else None

    # Try original
    encoding = get_encoding(img_path)
    if encoding is not None:
        return encoding

    # Try rotations: 90°, 180°, 270°
    for angle in [90, 180, 270]:
        rotate_image(img_path, angle)
        encoding = get_encoding(img_path)
        if encoding is not None:
            return encoding

    return None


def rotate_image(path, angle=90):
    """Rotate image by given angle clockwise"""
    try:
        im = Image.open(path)
        rotated_image = im.rotate(angle, expand=True)
        rotated_image.save(path)
    except Exception as e:
        frappe.log_error(str(e), "Rotate Image Error")

