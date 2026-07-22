# Prompt Bàn Giao Viết Bài Báo

Dùng prompt dưới đây khi đã có đủ số liệu (xem `journal_experiment_plan.md`
để biết trạng thái) để nhờ người/AI khác soạn bản thảo.

---

Bạn hãy soạn bản thảo bài báo tiếng Anh nộp cho IEEE Access (Scopus Q2) từ
một repository nghiên cứu vehicle Re-ID. Đây là paper dạng "dataset + method":
vừa công bố một dataset mới, vừa đề xuất một phương pháp huấn luyện mới
(WICV-Net) cải thiện kết quả trên dataset đó.

Trước khi viết, đọc các file sau trong repo để lấy đúng nội dung, không tự
bịa số liệu hay tuyên bố chưa được kiểm chứng:

- `docs/paper_brief.md` — định vị, đóng góp, tên đề xuất
- `docs/paper_template.md` — outline đầy đủ theo từng section, kèm chỉ dẫn
  lấy bảng/số liệu từ đâu
- `methods/wicv/README.md` — mô tả kỹ thuật của phương pháp và so sánh với
  literature hiện có
- `docs/dataset_statistics.md`, `docs/reid_split_statistics.md` — số liệu
  dataset
- Các file `results/*/summary.csv` và `eval.json` được liệt kê trong
  `docs/journal_experiment_plan.md` — số liệu thực nghiệm cuối cùng

Yêu cầu khi viết:

1. Bám sát outline trong `paper_template.md`, không tự thêm/bớt section.
2. Chỗ nào số liệu chưa có, giữ nguyên `TODO` thay vì bịa ra con số.
3. Văn phong học thuật, khách quan — không thổi phồng đóng góp; nếu có phát
   hiện bất lợi (ví dụ trọng số adversarial mặc định làm giảm hiệu năng),
   trình bày trung thực như một phần phân tích, không né tránh.
4. Giữ đúng thuật ngữ kỹ thuật đã dùng trong code (`CV-Tri`, `CVPA`, `FCA`,
   tên các baseline) để nhất quán với repository.
