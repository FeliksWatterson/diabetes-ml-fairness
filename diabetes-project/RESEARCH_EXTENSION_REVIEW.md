# Bảng kiểm tra 7 phần bổ sung theo research

Notebook: `notebooks/05_fairness.ipynb`. Cell đánh số theo thứ tự code cell, không phải số In[ ] của kernel.

| Phần | Nội dung | Code cell | Marker tìm kiếm | Hàm trong src/utils.py | File đầu ra |
|---|---|---:|---|---|---|
| 1 | AUC + CI 95% | 12 | `RESEARCH_AUC_CI` | `get_delong_auc_ci` | overall_auc_brier_ci.csv; group_metrics_with_ci.csv |
| 2 | Bảng III và delta AUC | 13 | `RESEARCH_TABLE_III` | `get_delong_auc_ci` | table_iii_auc_by_group.csv |
| 3 | Forest plot | 14 | `RESEARCH_FOREST_PLOT` | `get_delong_auc_ci` | forest_auc_2015.png; forest_auc_2017.png |
| 4 | Kiểm định AUC + Bonferroni | 15 | `RESEARCH_AUC_TESTS` | `get_subgroup_auc_tests; get_paired_auc_test` | auc_subgroup_tests_bonferroni.csv; auc_paired_model_tests.csv |
| 5 | Disparate impact | 16 | `RESEARCH_DISPARATE_IMPACT` | `get_disparate_impact` | disparate_impact_by_group.csv |
| 6 | Brier + CI 95% | 17 | `RESEARCH_BRIER_CI` | `get_brier_ci` | brier_ci_by_group.csv; overall_auc_brier_ci.csv |
| 7 | Decision Curve Analysis | 18 | `RESEARCH_DCA` | `get_decision_curve` | decision_curve_net_benefit.csv; decision_curve_beneficial_grid_ranges.csv; decision_curve_2015.png; decision_curve_2017.png |

## Nơi lưu kết quả

CSV/manifest: `results/fairness/54c463bbe9079503/research_extension/`.
Hình: `results/figures/fairness/54c463bbe9079503/research_extension/`.

## Phương pháp và giới hạn

- AUC: CI chuẩn pointwise từ phương sai DeLong. Brier: CI chuẩn trung bình squared loss, không phải bootstrap.
- Kiểm định nhóm độc lập, delta=nhóm-tham chiếu. Bonferroni 8 so sánh/năm/mô hình. XGBoost-LR paired DeLong, family riêng 2 năm.
- Tham chiếu: Male, White NH, 18-39, Normal. Missing/Underweight mô tả riêng. DI=selection rate nhóm/tham chiếu, không áp chuẩn pass/fail.
- DCA lưới định trước 0.01-0.80 bước 0.005 và ngưỡng đóng băng. Khoảng có lợi chỉ gồm điểm lưới, không tối ưu ngưỡng.
- Chưa xét survey weights/PSU/strata hay bất định huấn luyện. CI pooled outer OOF NHANES là có điều kiện, các dự đoán chia sẻ tập huấn luyện.
- DCA nhãn tự khai chưa chứng minh lợi ích triển khai thực tế. TPR/FPR và EO gap chưa có CI. Không recalibration/huấn luyện lại.

## Chạy lại

Chọn đúng kernel, chạy toàn bộ hoặc chạy từ RESEARCH_SETUP theo thứ tự. Không cần script bên ngoài.

## Nguồn phương pháp

https://doi.org/10.2307/2531595
https://doi.org/10.1177/0272989X06295361
