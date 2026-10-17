## Đã làm

- DEMO + DIQ, lọc tuổi ≥18 → tạo mẫu nghiên cứu, lấy thông tin chẩn đoán/thuốc để tạo nhãn.
- Kiểm tra SEQN: không trùng trong từng file → nối đúng người, không nhân dòng. Dùng `left merge`, `one_to_one` → giữ mẫu nền.
- Kiểm tra DIQ070 → phân biệt thiếu do bỏ qua câu hỏi với thiếu dù được hỏi. SEQN 91416 thuộc nhóm sau, vẫn có chẩn đoán dương tính.
- Nối HbA1c và glucose → bổ sung xét nghiệm để tạo nhãn, không đưa vào 8 biến dự đoán. Sau nối lần lượt có 13 và 15 cột.
- Đối chiếu FASTQX → kiểm tra điều kiện nhịn ăn trước khi dùng glucose. Nhóm có glucose nối đủ FASTQX, không thiếu giờ/phút, đều khám buổi sáng.
- Sửa `load_xpt()`: `2**-260` → 0 → tránh đếm nhầm trọng số 0 thành dương. Đã đối chiếu byte gốc GLU/FASTQX; giá trị khác và NaN giữ nguyên. DIQ vẫn đọc trực tiếp bằng `pd.read_sas`.
- Chạy đến ô 35; output notebook khớp kiểm tra Python riêng.

## Số liệu đã kiểm tra

| Chỉ tiêu | 2015–2016 | 2017–2018 |
|---|---:|---:|
| Người ≥18 tuổi | 5.992 | 5.856 |
| DIQ070 thiếu: do bỏ qua / dù được hỏi | 4.523 / 1 | 4.227 / 0 |
| Thiếu HbA1c | 551 | 595 |
| Có / thiếu glucose | 2.573 / 3.419 | 2.537 / 3.319 |
| Có glucose, trọng số dương / bằng 0 | 2.379 / 194 | 2.387 / 150 |
| Nhịn ăn <8 giờ, trong nhóm có glucose | 180 | 123 |
| Nhịn ăn 8–<9 giờ | 6 | 2 |
| Nhịn ăn 9–<24 giờ | 2.373 | 2.385 |
| Nhịn ăn ≥24 giờ | 14 | 27 |

- Ngoài 8–<24 giờ: 194 / 150 người, trùng nhóm trọng số 0 → có glucose chưa đồng nghĩa đủ điều kiện sử dụng.
- Bảng cũ ghi không có trọng số 0: sai do lỗi đọc XPT. Đã sửa bằng số phía trên.
- Giữ NaN, chưa loại người → thiếu xét nghiệm chưa có nghĩa âm tính; nhãn còn thành phần khác. Dữ liệu gốc giữ nguyên.

## Chưa làm

- Kiểm tra ăn/uống bổ sung trong FASTQX → hoàn tất kiểm tra nhịn ăn.
- Làm rõ mốc <9 giờ ở GLU I và 8–<24 giờ ở GLU J → chốt điều kiện dùng glucose.
- Chốt cách xử lý thiếu khi tạo nhãn; tạo nhãn cuối cùng.
- Thu thập đủ 8 biến dự đoán; xử lý BRFSS; chạy mô hình.

## So với bài báo

- Chọn giai đoạn trước COVID: NHANES 2015–2018, BRFSS 2015–2018. Hai nguồn dùng riêng cho phát triển / kiểm thử ngoài.
- FASTQX: bước kiểm tra dữ liệu bổ sung theo CDC.
- PDF `2607.16253v1.pdf`: chưa đọc → chưa xác minh khác biệt phương pháp còn lại.

Nguồn: [GLU I](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GLU_I.htm), [GLU J](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm), [FASTQX](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/FASTQX_J.htm).
