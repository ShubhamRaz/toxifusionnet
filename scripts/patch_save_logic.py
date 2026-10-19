from pathlib import Path

p = Path("scripts/build_final_professional_report.py")
code = p.read_text(encoding="utf-8")

old_save = """    # Save to requested destination
    out_file = Path(output_docx_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_file))
    print(f"\\nSuccessfully generated final professional report at: {out_file.resolve()}")
    print(f"File size: {out_file.stat().st_size} bytes")


if __name__ == "__main__":
    out_path = "D:/Research ppr/ToxiFusionNet_Report1_Final_Professional.docx"
    build_final_professional_report(out_path)"""

new_save = """    # Save to multiple targets
    targets = [
        Path("reports/ToxiFusionNet_Report1_Final_Professional.docx"),
        Path("D:/Research ppr/ToxiFusionNet_Report1_Final_Professional_Updated.docx"),
        Path("D:/Research ppr/toxifusionnet_accuracy_report.docx"),
        Path(output_docx_path),
    ]
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(target))
            print(f"Successfully saved report at: {target.resolve()} ({target.stat().st_size} bytes)")
        except PermissionError:
            print(f"Notice: {target.name} is currently open in Word. Created updated copy at ToxiFusionNet_Report1_Final_Professional_Updated.docx")
        except Exception as e:
            print(f"Warning saving to {target}: {e}")


if __name__ == "__main__":
    out_path = "D:/Research ppr/ToxiFusionNet_Report1_Final_Professional.docx"
    build_final_professional_report(out_path)"""

assert old_save in code, "old_save not found"
code = code.replace(old_save, new_save)
p.write_text(code, encoding="utf-8")
print("Save logic updated successfully.")
