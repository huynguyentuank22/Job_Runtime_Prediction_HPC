import fitz  # Thư viện PyMuPDF
import os
import glob

def convert_pdfs_to_png(directory):
    # Tìm tất cả các file .pdf trong thư mục
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    
    if not pdf_files:
        print("Không tìm thấy file PDF nào trong thư mục.")
        return

    print(f"Tìm thấy {len(pdf_files)} file PDF. Bắt đầu chuyển đổi...\n")

    for pdf_path in pdf_files:
        print(f"Đang xử lý: {os.path.basename(pdf_path)}")
        try:
            # Mở file PDF
            doc = fitz.open(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for i, page in enumerate(doc):
                # Thiết lập độ phân giải 300 DPI để ảnh PNG có chất lượng tốt
                pix = page.get_pixmap(dpi=300)
                
                # Định dạng tên file đầu ra
                if len(doc) == 1:
                    png_path = os.path.join(directory, f"{base_name}.png")
                else:
                    png_path = os.path.join(directory, f"{base_name}_page_{i+1}.png")
                
                # Lưu ảnh dưới dạng PNG
                pix.save(png_path)
                print(f"  -> Đã lưu: {os.path.basename(png_path)}")
                
        except Exception as e:
            print(f"Lỗi khi xử lý {os.path.basename(pdf_path)}: {e}")

if __name__ == "__main__":
    # Đường dẫn tới thư mục chứa ảnh
    images_dir = "/home/dangbao/Desktop/WorkSpace/Research HPC/Job_Runtime_Prediction_HPC_new_1_1/images"
    
    # Kiểm tra xem thư mục có tồn tại không
    if os.path.exists(images_dir):
        convert_pdfs_to_png(images_dir)
        print("\nHoàn tất chuyển đổi!")
    else:
        print(f"Thư mục không tồn tại: {images_dir}")
