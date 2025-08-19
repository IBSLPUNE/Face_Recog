import cv2
import frappe
@frappe.whitelist()
def after_insert_face_capture(snapshot_file_url):
    """Hook: Triggered when a new Face Capture is inserted"""
    if not snapshot_file_url:
        return

    snap_file_doc = frappe.get_doc("File", {"file_url": snapshot_file_url})
    snap_img_path = snap_file_doc.get_full_path()
    #frappe.throw(str(snap_img_path))

    # Compare with all employees (or specific one if provided)
    employees = frappe.get_all("Employee", filters={"image": ["!=", ""]}, fields=["name", "employee_name", "image"])

    for emp in employees:
        emp_file_doc = frappe.get_doc("File", {"file_url": emp.image})
        emp_img_path = emp_file_doc.get_full_path()
        #frappe.throw(str(emp_img_path))

        if _faces_match(emp_img_path, snap_img_path):
            # Update Face Capture status
            #doc.status = "Success"
            #doc.employee = emp.name
            #frappe.throw(str(emp))            #frappe.db.set_value("Face Capture", doc.name, {"status": "Success", "employee": emp.name})

            # Auto create Checkin
            checkin = frappe.get_doc({
                "doctype": "Employee Checkin",
                "employee": emp.name,
                "log_type": "IN"
            })
            checkin.insert(ignore_permissions=True)

            # Attach the face snapshot to Checkin
            frappe.get_doc({
                "doctype": "File",
                "file_url": snapshot_file_url,   # reuse Face Capture uploaded file
                "attached_to_doctype": "Employee Checkin",
                "attached_to_name": checkin.name,
                "attached_to_field": None,       # optional
                "is_private": 1
            }).insert(ignore_permissions=True)
            return

    # If no match found
    #doc.status = "Failed"
    #frappe.db.set_value("Face Capture", doc.name, "status", "Failed")


def _faces_match(ref_img_path, test_img_path):
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(test_img, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    good_matches = [m for m in matches if m.distance < 50]  # threshold tuning
    return len(good_matches) > 20

