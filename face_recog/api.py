import cv2
import frappe
import base64
from frappe.utils.file_manager import save_file

@frappe.whitelist()
def after_insert_face_capture(snapshot_file_url):
    """Triggered when a new Face Capture is inserted via base64 snapshot"""
    if not snapshot_file_url:
        return

    # Extract base64 image data
    base64_data = snapshot_file_url.split(",")[1]
    image_data = base64.b64decode(base64_data)

    # Save snapshot as File Doc
    file_name = f"face_capture_{frappe.utils.now_datetime().strftime('%Y%m%d%H%M%S')}.jpg"
    _file = save_file(
        file_name,
        image_data,
        decode=False,
        is_private=True
    )
    snap_file_doc = frappe.get_doc("File", {"file_url": _file.file_url})
    snap_img_path = snap_file_doc.get_full_path()

    # Compare with all employees who have images
    employees = frappe.get_all(
        "Employee",
        filters={"image": ["!=", ""]},
        fields=["name", "employee_name", "image"]
    )

    for emp in employees:
        emp_file_doc = frappe.get_doc("File", {"file_url": emp.image})
        emp_img_path = emp_file_doc.get_full_path()

        if _faces_match(emp_img_path, snap_img_path):
            # Auto create Checkin
            checkin = frappe.get_doc({
                "doctype": "Employee Checkin",
                "employee": emp.name,
                "log_type": "IN"
            })
            checkin.insert(ignore_permissions=True)

            # Attach the saved file (not base64)
            frappe.get_doc({
                "doctype": "File",
                "file_url": _file.file_url,
                "attached_to_doctype": "Employee Checkin",
                "attached_to_name": checkin.name,
                "attached_to_field": None,
                "is_private": 1
            }).insert(ignore_permissions=True)

            return {"status": "success", "employee": emp.name}

    return {"status": "failed"}


def _faces_match(ref_img_path, test_img_path):
    """Compare faces using ORB feature matching"""
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or test_img is None:
        return False

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(test_img, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower = better)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    # Compare relative threshold
    similarity = len(good_matches) / max(len(matches), 1)
    return similarity > 0.3   # 30% match threshold

