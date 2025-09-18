import frappe
import json
import face_recognition
import numpy as np

@frappe.whitelist(allow_guest=True)
def save_face_embedding(image_base64):
    """
    Auto check-in employee by comparing face with stored embeddings.
    - image_base64: base64 encoded image from frontend
    """
    frappe.flags.allow_guest = True
    frappe.flags.ignore_csrf = True

    import base64
    from io import BytesIO
    from PIL import Image

    try:
        if not image_base64:
            frappe.throw("No image received")

        # Decode image
        image_data = base64.b64decode(image_base64.split(",")[-1])
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img_np = np.array(img)

        # Extract face embedding
        encodings = face_recognition.face_encodings(img_np)
        if not encodings:
            frappe.throw("No face found in image")

        embedding = encodings[0]

        # Loop over all enrolled employees
        employees = frappe.get_all("Employee", fields=["name", "employee_name", "custom_face_embedding"])
        matched_emp = None
        min_dist = 1e9

        for emp in employees:
            if not emp.custom_face_embedding:
                continue

            stored_embedding = np.array(json.loads(emp.custom_face_embedding))
            distance = np.linalg.norm(embedding - stored_embedding)

            if distance < 0.6 and distance < min_dist:
                min_dist = distance
                matched_emp = emp

        if not matched_emp:
            frappe.throw("No matching employee found. Please enroll first.")

        # Create check-in record
        frappe.get_doc({
            "doctype": "Employee Checkin",
            "employee": matched_emp.name,
            "log_type": "IN"
        }).insert(ignore_permissions=True)
        frappe.db.commit()

        return {"status": "success", "message": f"✅ Check-in successful for {matched_emp.employee_name}"}

    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Face Auto Checkin Error")
        return {"status": "error", "message": str(e)}


# Utility to calculate embeddings manually (use in console to store in Employee Doctype)
@frappe.whitelist()
def calculate_face_embedding(image_base64):
    """
    Calculate face embedding and return array for manual storage in Employee Doctype.
    """
    import base64
    from io import BytesIO
    from PIL import Image

    image_data = base64.b64decode(image_base64.split(",")[-1])
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img_np = np.array(img)

    encodings = face_recognition.face_encodings(img_np)
    if not encodings:
        frappe.throw("No face found in image")

    return encodings[0].tolist()
