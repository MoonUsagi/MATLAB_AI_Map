# MATLAB AI 學習地圖

一份以 MATLAB 為核心的人工智慧學習資源索引，涵蓋影像分類、物件偵測、影像分割、異常偵測、OCR、訊號與音訊、LLM、Python 整合、程式部署，以及低程式碼 App。

本專案將教學影片、文章與範例程式集中整理成一條可循序學習的路徑。你可以從預訓練模型與推論開始，再逐步進入資料標註、遷移學習、模型訓練、可解釋 AI、跨語言整合與邊緣裝置部署。

> [!NOTE]
> 部分資源建立於較早的 MATLAB 版本，介面、函式名稱或模型支援情況可能因版本而異。實作前請確認影片或文章所標示的版本與所需 Toolbox。

## 適合對象

- 想以 MATLAB 入門機器學習、深度學習或電腦視覺的學習者
- 需要快速找到 AI 範例、教學影片與程式碼的工程師
- 從影像處理延伸至物件偵測、影像分割或視覺檢測的開發者
- 想整合 MATLAB、Python、TensorFlow、PyTorch 或 ONNX 的使用者
- 準備將模型部署至桌面程式、GPU 或 Jetson 裝置的團隊

## 建議學習路徑

1. **快速入門**：下載預訓練模型，完成第一次影像分類推論。
2. **模型訓練**：學習資料整理、標註、遷移學習與從頭訓練。
3. **進階視覺任務**：依需求進入物件偵測、語意分割、實例分割或異常偵測。
4. **模型理解**：使用 Grad-CAM、LIME、Occlusion 等方法解讀預測結果。
5. **系統整合**：串接 Python、封裝應用程式，或產生可部署程式碼。
6. **硬體部署**：將模型部署至 GPU、Jetson Nano、ARM 或 FPGA。

## 目錄

- [MATLAB 版本與 AI 更新](#matlab-版本與-ai-更新)
- [影像分類](#影像分類)
- [物件偵測](#物件偵測)
- [語意分割](#語意分割)
- [實例分割](#實例分割)
- [模型推論與展示](#模型推論與展示)
- [文字偵測、OCR 與條碼](#文字偵測ocr-與條碼)
- [異常偵測與自動光學檢測](#異常偵測與自動光學檢測)
- [追蹤、動作估測與姿態估測](#追蹤動作估測與姿態估測)
- [數值、訊號與音訊](#數值訊號與音訊)
- [大型語言模型](#大型語言模型)
- [深度學習延伸主題](#深度學習延伸主題)
- [MATLAB 與 Python 整合](#matlab-與-python-整合)
- [軟體整合與程式碼產生](#軟體整合與程式碼產生)
- [硬體整合與部署](#硬體整合與部署)
- [低程式碼與圖形化 App](#低程式碼與圖形化-app)
- [自製 App 與延伸專案](#自製-app-與延伸專案)
- [相關程式碼專案](#相關程式碼專案)
- [未來規劃](#未來規劃)

---

## MATLAB 版本與 AI 更新

整理各版本與 AI、影像、訊號及 LLM 相關的重要更新，適合用來快速掌握新功能與既有工作流程的差異。

| 版本 | 類型 | 資源 |
| --- | --- | --- |
| R2024a | 影片 | [MATLAB R2024a AI 功能更新](https://youtu.be/RA6n_7yd40E?si=_lT2-ISEkYDt2VF6) |
| R2024a | 文章 | [MATLAB R2024a AI Update：影像](https://medium.com/@FredLiu_/matlab-2024a-ai-update-%E5%BD%B1%E5%83%8F-30abf48c7e12) |
| R2024a | 文章 | [MATLAB R2024a AI Update：LLM、訊號、音訊與通訊](https://medium.com/@FredLiu_/matlab-2024a-ai-update-llms-%E8%A8%8A%E8%99%9F%E9%9F%B3%E8%A8%8A%E9%80%9A%E8%A8%8A-a05118f46ec8) |
| R2023b | 影片 | [MATLAB R2023b AI 功能更新](https://youtu.be/TAoXGqqNzek?si=uyLRfFz9cunVy1yG) |
| R2023b | 文章 | [MATLAB R2023b AI Update：影像與電腦視覺](https://medium.com/@FredLiu_/matlab-2023b-ai-update-%E4%B8%AD-105d651190d0) |
| R2023b | 文章 | [MATLAB R2023b AI Update：Team Labeler](https://medium.com/@FredLiu_/matlab-2023b-ai-update-%E4%B8%8A-ac6cd8012066) |
| R2023b | 文章 | [MATLAB R2023b AI Update：深度學習、機器學習、訊號與音訊](https://medium.com/@FredLiu_/matlab-2023b-ai-update-deep-learning-machine-learing-signal-audio-36b9f175aa57) |

[回到目錄](#目錄)

---

## 影像分類

影像分類是進入深度學習最直觀的起點。本章從預訓練模型、少量程式碼推論與圖形化工具開始，再延伸至多張影像分類、遷移學習、自訂模型訓練，以及模型判斷依據的視覺化。

### 快速入門

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| 安裝預訓練深度學習模型 | [在 MATLAB 中安裝預訓練深度學習模型](https://youtu.be/ZPCNmTxV5K8) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 使用少量程式碼完成影像分類 | [用五行 MATLAB 程式完成影像分類](https://www.youtube.com/watch?v=fjjnGmvZxc0) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 使用 Transfer Learning App | [使用 Transfer Learning App 完成遷移學習](https://youtu.be/7qlJgBoSnKA) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 使用 Deep Network Designer | [使用 Deep Network Designer 建立深度學習模型](https://youtu.be/AQw3DC7FK1Y) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |

### 分類與遷移學習

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| 影像分類（一） | [MATLAB 深度學習影像分類（一）](https://youtu.be/5kvmg2uCpdE) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 影像分類（二） | [MATLAB 深度學習影像分類（二）](https://youtu.be/nNCa8rU5Jms) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 影像分類（三） | [MATLAB 深度學習影像分類（三）](https://youtu.be/Yg8hlyjPO5Q) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 遷移學習（一） | [MATLAB 深度學習遷移學習（一）](https://youtu.be/v0ZwOiQhpi4) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 遷移學習（二） | [MATLAB 深度學習遷移學習（二）](https://youtu.be/YUY2KhgGWuw) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 遷移學習（三） | [MATLAB 深度學習遷移學習（三）](https://youtu.be/s8JfMbs_CSw) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |

### 模型視覺化與可解釋 AI

這些方法可協助判斷模型在分類時關注的影像區域，並用於除錯、偏誤檢查與模型驗證。

| 方法 | 教學影片 | 範例程式 |
| --- | --- | --- |
| 網路與特徵視覺化 | [MATLAB 深度學習網路與特徵視覺化](https://youtu.be/XpnCzsBvMQQ) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| DeepDream | [使用 DeepDream 視覺化模型特徵](https://youtu.be/zDbv-fNAvn4) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| Grad-CAM | [使用 Grad-CAM 解讀分類結果](https://youtu.be/30t-ARZNDjA) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| Gradient Attribution | [使用 Gradient Attribution 解讀模型](https://youtu.be/cliegS5uZuc) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| LIME | [使用 LIME 解讀分類模型](https://youtu.be/SsUwRYGRt7E) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| Occlusion Sensitivity | [使用 Occlusion Sensitivity 分析模型](https://youtu.be/AdV-Ii0hfxM) | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |

[回到目錄](#目錄)

---

## 物件偵測

物件偵測同時預測物件類別與位置。本章涵蓋影像標註、資料準備、模型訓練、雲端運算、推論與部署，並比較 Faster R-CNN、SSD、YOLOv2、YOLOv3、YOLOv4 與 YOLOX 等模型。

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| 物件偵測入門（一） | [MATLAB 深度學習物件偵測入門（一）](https://youtu.be/ISzFfL-W9AE) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| 物件偵測入門（二）：YOLOX | [MATLAB 深度學習物件偵測入門（二）：YOLOX](https://youtu.be/d1IJS6CYvQw?si=dvk3G71ll9L3Qoso) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| RabbitDetect：資料準備 | [RabbitDetect 物件偵測：資料準備](https://youtu.be/6gHTFQeD8Xw) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| RabbitDetect：標註與訓練流程 | [RabbitDetect 物件偵測：標註與訓練流程](https://youtu.be/g9S0_VSfkFQ) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| Faster R-CNN | [RabbitDetect 物件偵測：Faster R-CNN](https://youtu.be/uXkvQup0pe0) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| SSD | [RabbitDetect 物件偵測：SSD](https://youtu.be/VxssEJBObas) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOv2 | [RabbitDetect 物件偵測：YOLOv2](https://youtu.be/VC5pRCv_QCo) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOv3 | [RabbitDetect 物件偵測：YOLOv3](https://youtu.be/CT8diNnOkXs) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOv4 | [RabbitDetect 物件偵測：YOLOv4](https://youtu.be/glXUnqScaGc) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOv4 延伸實作 | [RabbitDetect 物件偵測：YOLOv4 延伸實作](https://youtu.be/Wzes49qzwCM) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOX | [RabbitDetect 物件偵測：YOLOX](https://youtu.be/zEJurTM1PUI?si=AS9Qav1JjCrh5Rlb) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |

### 延伸資源

| 主題 | 資源 |
| --- | --- |
| 在 MATLAB 中使用 YOLOv4 | [從 YOLOv4 論文到 MATLAB 實作](https://youtu.be/hJrZg94y8vA) |
| 使用 TWCC 雲端資源訓練 AI 模型 | [使用 TWCC 雲端資源訓練 AI 模型](https://youtu.be/MZcEBpZFVwg) |
| 透過 Object Detection App 使用 YOLOv4 | [無程式碼物件偵測：Object Detection App 與 YOLOv4](https://youtu.be/I3cWtTl3b3A) |
| YOLOX 推論 | [YOLOX Inference](https://youtu.be/TDf8SHRFrCU?si=ik0x6Wb3dwT4jamz) |
| YOLOX 與 MATLAB R2023b | [閱讀文章](https://medium.com/@FredLiu_/yolox-matlab-2023b-1987f2b3aa05) |

[回到目錄](#目錄)

---

## 語意分割

語意分割會為每個像素預測類別，適合表面缺陷、醫療影像、道路場景與遙測影像等任務。相較於物件偵測，像素級標註的成本與運算需求通常更高。本章以 DeepLabv3+ 的資料準備、訓練與推論流程為主。

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| RabbitDetect 語意分割：資料與標註 | [RabbitDetect 語意分割：資料與標註](https://youtu.be/ZKXTZ0RCYWg) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| RabbitDetect 語意分割：訓練流程 | [RabbitDetect 語意分割：訓練流程](https://youtu.be/11DiP35W-dg) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| DeepLabv3+（上） | [RabbitDetect 語意分割：DeepLabv3+（上）](https://youtu.be/v02Np1_q08o) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| DeepLabv3+（下） | [RabbitDetect 語意分割：DeepLabv3+（下）](https://youtu.be/LzbrKeMOnDY) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |

[回到目錄](#目錄)

---

## 實例分割

實例分割不只區分像素類別，也會識別同類別中的不同物件個體。本章以 Mask R-CNN 為核心，介紹推論、標註、資料準備與模型訓練。

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| Mask R-CNN 推論 | [RabbitDetect 實例分割：Mask R-CNN 推論](https://www.youtube.com/watch?v=EG78LZkvREM) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| Mask R-CNN 資料準備與推論 | [RabbitDetect 實例分割：資料準備與推論](https://youtu.be/sPa4UHyq65Y) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| Mask R-CNN 標註與訓練流程 | [RabbitDetect 實例分割：標註與訓練流程](https://youtu.be/6nNKu2qjWcQ) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| Mask R-CNN 訓練 | [RabbitDetect 實例分割：訓練 Mask R-CNN](https://youtu.be/ehov2aTgesU) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| Mask R-CNN 訓練結果 | [RabbitDetect 實例分割：Mask R-CNN 訓練結果](https://youtu.be/_lNHy4jLp00) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |

[回到目錄](#目錄)

---

## 模型推論與展示

本節集中展示不同模型的推論流程，適合先確認模型輸入、輸出與執行結果，再回頭理解訓練細節。

| 模型或主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| YOLOX：R2023b 功能更新 | [MATLAB R2023b 深度學習推論：YOLOX](https://youtu.be/KBQc0f5fH58?si=T3RVVFImWaJEuZkB) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOX：推論展示 | [YOLOX 推論展示](https://youtu.be/TDf8SHRFrCU?si=GXv1zqN8hOqPBPe3) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| SOLOv2：R2023b 功能更新 | [MATLAB R2023b 深度學習推論：SOLOv2](https://youtu.be/CqLdOKNson8?si=5f3Mb5LEDXy6i71X) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| YOLOX：單張影像推論 | [YOLOX 單張影像推論](https://youtu.be/U-9MKdPHaac?si=sLU0Om4EG59St8dL) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| SOLOv2：單張影像推論 | [SOLOv2 單張影像推論](https://youtu.be/E0wst5Dxs2U?si=ld9T8RNAizuTb_IS) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| HRNet 推論 | [HRNet 深度學習推論](https://youtu.be/NEPINAMEjpM?si=lfRjnfV8fT2AuP4a) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |

[回到目錄](#目錄)

---

## 文字偵測、OCR 與條碼

這些技術常見於工業自動化、文件處理、產品追溯與品質檢測。本節涵蓋文字區域偵測、深度學習式 OCR，以及一維與二維條碼辨識。

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| Text Detection | [MATLAB 影像文字區域偵測（Text Detection）](https://youtu.be/fOl85S2SSw0) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| 深度學習式 OCR | [MATLAB 深度學習式 OCR](https://youtu.be/bZVOEviIcQE) | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| Barcode 辨識 | 待補 | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |

[回到目錄](#目錄)

---

## 異常偵測與自動光學檢測

異常偵測適合缺陷樣本稀少、類別不平衡，或無法事先列舉所有瑕疵型態的情境。相關方法可應用於影像、訊號與數值資料；本節聚焦工業視覺檢測與 AOI 實作。

| 類型 | 資源 |
| --- | --- |
| 範例程式 | [AOI_Lab](https://github.com/MoonUsagi/AOI_Lab/tree/main) |
| 文章 | [AOI Lab：MATLAB Visual Inspection（上）](https://medium.com/@FredLiu_/aoi-lab-matlab-visual-inspection-%E4%B8%8A-524cd52fc939) |
| 文章 | [AOI Lab：MATLAB Visual Inspection（下）](https://medium.com/@FredLiu_/aoi-lab-matlab-visual-inspection-%E4%B8%8B-5d35f7a5a0af) |

[回到目錄](#目錄)

---

## 追蹤、動作估測與姿態估測

本節介紹影像序列中的目標追蹤與運動估測，並延伸至人體姿態估測與即時裝置部署。

| 主題 | 資源 |
| --- | --- |
| Tracking and Motion Estimation（一） | [Tracking and Motion Estimation（一）](https://youtu.be/tzncUlQfgMs?si=KbrGyPUZKfMRP48M) |
| Tracking and Motion Estimation（二） | [Tracking and Motion Estimation（二）](https://youtu.be/xENfpLsNwh0?si=yLz8jnZ8pISY4Wva) |
| Human Pose Estimation with Jetson Nano | [Human Pose Estimation with Jetson Nano](https://www.youtube.com/watch?v=_ACXj8rrBtw) |

[回到目錄](#目錄)

---

## 數值、訊號與音訊

深度學習不限於影像資料，也可用於時間序列、感測器訊號、音訊與一般數值資料。本節提供對應的教學文章與程式範例。

| 類型 | 資源 |
| --- | --- |
| 教學文章 | [深度學習數值：DL Num](https://medium.com/@FredLiu_/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%95%B8%E5%80%BC-dl-num-a838c1e41ad2) |
| 範例程式 | [DL_Num](https://github.com/MoonUsagi/DL_Num) |

[回到目錄](#目錄)

---

## 大型語言模型

介紹如何在 MATLAB 工作流程中使用大型語言模型（LLM），包含開源範例，以及 MATLAB 與 ChatGPT 的互動情境。

| 類型 | 資源 |
| --- | --- |
| 影片 | [LLMs with MATLAB](https://youtu.be/reYhAnXMKRU?si=ltyGintjB16XidnD) |
| 影片 | [在 ChatGPT 中使用 MATLAB](https://youtu.be/ANYWtf1olYg?si=gudbGp6p7wE0uPHu) |
| 範例程式 | [Large Language Models with MATLAB](https://github.com/matlab-deep-learning/llms-with-matlab) |

[回到目錄](#目錄)

---

## 深度學習延伸主題

收錄不屬於單一任務類型的進階範例與延伸學習材料。

| 類型 | 資源 |
| --- | --- |
| 教學文章 | [深度學習擴充：DL Exten](https://medium.com/@FredLiu_/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%93%B4%E5%85%85-dl-exten-70f7f559a443) |
| 影片 | [DL Exten 專案介紹](https://youtu.be/XQqgnzd3KRs?si=x06bG3AtCfASd9nO) |
| 範例程式 | [DL_Exten](https://github.com/MoonUsagi/DL_Exten) |

[回到目錄](#目錄)

---

## MATLAB 與 Python 整合

MATLAB 與 Python 可以雙向呼叫，也能交換 TensorFlow、PyTorch 與 ONNX 模型。這類整合適合沿用既有 Python 生態，同時使用 MATLAB 進行資料處理、視覺化、演算法驗證或部署。

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| TensorFlow、PyTorch 與 ONNX 模型整合 | [MATLAB Integration（一）：TensorFlow、PyTorch 與 ONNX](https://www.youtube.com/watch?v=zlpyDuOIsLs) | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |
| 在 MATLAB 中執行 Python | [MATLAB Integration（二）：在 MATLAB 中執行 Python](https://www.youtube.com/watch?v=KBsTRPpvo3M) | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |
| 在 Python 中呼叫 MATLAB | [MATLAB Integration（三）：在 Python 中呼叫 MATLAB](https://www.youtube.com/watch?v=2H57hKkQevE) | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |

延伸閱讀：

- [MATLAB 與 Python、TensorFlow、PyTorch 整合](https://medium.com/@FredLiu_/%E5%AF%A6%E7%8F%BE%E6%95%B4%E5%90%88%E7%B3%BB%E5%88%97%E7%AC%AC%E4%B8%80%E6%9C%9F-matlab%E4%B8%AD%E5%AF%ABpython-%E8%88%87tensorflow-pytorch%E6%95%B4%E5%90%88-40b962bdc610)
- [MATLAB_Integration_Python（Tim）](https://github.com/sitdownplz/MATLAB_Integration_Python)

[回到目錄](#目錄)

---

## 軟體整合與程式碼產生

MATLAB 提供多種部署方式：

- **MATLAB Compiler**：將 MATLAB 應用程式封裝給沒有 MATLAB 的使用者執行。
- **MATLAB Compiler SDK**：建立可由 C/C++、.NET、Java 或 Python 等環境呼叫的元件。
- **MATLAB Coder**：從 MATLAB 演算法產生 C/C++ 程式碼。
- **GPU Coder**：產生可在 NVIDIA GPU 上執行的 CUDA 程式碼。

### Compiler 與 Compiler SDK

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| MATLAB Compiler | [MATLAB Integration（四）：MATLAB Compiler](https://youtu.be/CphzcqYFVH4?si=q6X3rYqMyzhEhwJL) | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |
| MATLAB Compiler SDK（上） | [MATLAB Integration（五）：Compiler SDK（上）](https://youtu.be/g3l5AXdRfPE?si=OQcnzLDb9nSpv4J1) | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |
| MATLAB Compiler SDK（下） | [MATLAB Integration（六）：Compiler SDK（下）](https://youtu.be/P0W8z_LtQzM?si=DLoMcJzjmDgXZdDw) | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |

延伸閱讀：

- [MATLAB 軟體整合架構介紹](https://medium.com/@FredLiu_/matlab%E5%AF%A6%E7%8F%BE%E6%95%B4%E5%90%88%E7%B3%BB%E5%88%97-%E6%A1%86%E6%9E%B6%E4%BB%8B%E7%B4%B9-93dfaa228a73)
- [MATLAB Compiler](https://medium.com/@FredLiu_/matlab%E5%AF%A6%E7%8F%BE%E6%95%B4%E5%90%88%E7%B3%BB%E5%88%97%E7%AC%AC%E4%BA%8C%E6%9C%9F-compiler-1c8ecc951ab0)
- [MATLAB Compiler SDK：C#/.NET](https://medium.com/@FredLiu_/matlab%E5%AF%A6%E7%8F%BE%E6%95%B4%E5%90%88%E7%B3%BB%E5%88%97%E7%AC%AC%E4%B8%89%E6%9C%9F-compiler-sdk-%E4%B8%8A-c-net%E7%92%B0%E5%A2%83%E7%82%BA%E4%BE%8Bc-ecbc1cdca022)
- [MATLAB Compiler SDK：C++](https://medium.com/@FredLiu_/matlab%E5%AF%A6%E7%8F%BE%E6%95%B4%E5%90%88%E7%B3%BB%E5%88%97%E7%AC%AC%E4%B8%89%E6%9C%9F-compiler-sdk-%E4%B8%8B-c-%E7%92%B0%E5%A2%83%E7%82%BA%E4%BE%8B-3ff0339077a3)

### GPU Coder

| 類型 | 資源 |
| --- | --- |
| 影片 | [GPU Coder 教學](https://youtu.be/dNSdl5UEucw?si=YoLBy14JrvGa4GCl) |
| 文章 | [MATLAB GPU Coder（上）](https://medium.com/@FredLiu_/matlab%E5%AF%A6%E7%8F%BE%E6%95%B4%E5%90%88%E7%B3%BB%E5%88%97%E7%AC%AC%E5%9B%9B%E6%9C%9F-gpu-coder-%E4%B8%8A-5494492adcd4) |

[回到目錄](#目錄)

---

## 硬體整合與部署

本節聚焦將 MATLAB AI 模型部署到 NVIDIA Jetson Nano，並保留 ARM 與 FPGA 的後續擴充方向。

| 主題 | 教學影片 | 範例程式 |
| --- | --- | --- |
| 使用 GPU Coder 將 YOLO 部署至 Jetson Nano | [MATLAB GPU Coder：將 YOLO 部署至 Jetson Nano](https://www.youtube.com/watch?v=uBWmpXLGyXE) | [MATLAB_Mask_Detection-with-Jetoson-Nano](https://github.com/MoonUsagi/MATLAB_Mask_Detection-with-Jetoson-Nano) |
| Jetson Nano 人體姿態估測 | [使用 Jetson Nano 進行人體姿態估測](https://www.youtube.com/watch?v=_ACXj8rrBtw) | [MATLAB_Mask_Detection-with-Jetoson-Nano](https://github.com/MoonUsagi/MATLAB_Mask_Detection-with-Jetoson-Nano) |
| ResNet-50 on Jetson Nano | — | [Jetson_Nano_resnet50](https://github.com/MoonUsagi/Jetson_Nano_resnet50) |

[回到目錄](#目錄)

---

## 低程式碼與圖形化 App

MATLAB App 可協助使用者以視覺化介面完成資料標註、影像分割、相機校正、模型設計與實驗管理。多數 App 也能產生 MATLAB 程式碼，便於把互動式流程轉成可重複執行的腳本。

### 影像處理與電腦視覺 App

| App 或主題 | 教學影片 |
| --- | --- |
| Image Labeler（R2023a 更新） | [MATLAB R2023a 更新：Image Labeler](https://youtu.be/jghZbfJo3j8) |
| Image Region Analyzer（R2023a 更新） | [MATLAB R2023a 更新：Image Region Analyzer](https://youtu.be/FsArD6K0ong) |
| Color Thresholder（一） | [MATLAB R2023a 更新：Color Thresholder（一）](https://youtu.be/vJDSWuhr40k) |
| Color Thresholder（二） | [MATLAB R2023a 更新：Color Thresholder（二）](https://youtu.be/6ihlgcRin1A) |
| Image Segmenter（R2023a 更新） | [MATLAB R2023a 更新：Image Segmenter](https://youtu.be/4QKOlREl8ZI) |
| Color Thresholder | [使用 Color Thresholder 進行色彩分割](https://youtu.be/vPB9dl8lMvw) |
| Image Segmenter | [使用 Image Segmenter 進行影像分割](https://youtu.be/kumg3rujj3U) |
| Image Region Analyzer | [使用 Image Region Analyzer 分析影像區域](https://youtu.be/4T-zSLD8Eos) |
| Registration Estimator | [使用 Registration Estimator 進行影像對位](https://youtu.be/dIP6juyMYFQ) |
| Image Labeler | [使用 Image Labeler 標註影像](https://youtu.be/Tq7f_6NOjEU) |
| Image Acquisition | [使用 Image Acquisition 取得影像](https://youtu.be/OSY7CdH4w2g) |
| Image Acquisition 延伸教學 | [Image Acquisition 延伸教學](https://youtu.be/b_UgBJZC4XY) |
| Image Batch Processor | [使用 Image Batch Processor 批次處理影像](https://youtu.be/RSI86ZFnzsI) |
| Hyperspectral Viewer | [使用 Hyperspectral Viewer 檢視高光譜影像](https://youtu.be/vaMoSDypyX4) |

### 3D 與醫療影像 App

| App | 教學影片 |
| --- | --- |
| Volume Viewer | [使用 Volume Viewer 檢視三維影像](https://youtu.be/9gALqxyHKsI) |
| Volume Segmenter | [使用 Volume Segmenter 分割三維影像](https://youtu.be/0KZGW29FuBA) |
| Medical Image Labeler | [使用 Medical Image Labeler 標註醫療影像](https://youtu.be/Obj8I07mXuY) |

### LiDAR 與相機校正 App

| App | 教學影片 |
| --- | --- |
| Lidar Viewer | [使用 Lidar Viewer 檢視點雲資料](https://youtu.be/87VnsitFVCI) |
| Lidar Camera Calibrator | [使用 Lidar Camera Calibrator 進行感測器校正](https://youtu.be/_WabP7g21kM) |
| Camera Calibrator | [使用 Camera Calibrator 進行相機校正](https://youtu.be/U6JfjgITDrs) |
| Stereo Camera Calibrator | [使用 Stereo Camera Calibrator 進行雙目相機校正](https://youtu.be/wRp18LrY_5k) |
| Lidar Labeler | [使用 Lidar Labeler 標註點雲資料](https://youtu.be/CurDmG9rYbI) |

### AI 與深度學習 App

| App 或主題 | 教學影片 |
| --- | --- |
| Reinforcement Learning Designer | [使用 Reinforcement Learning Designer 建立強化學習流程](https://youtu.be/4G2LHFJvR2E) |
| Deep Network Designer（R2022a 更新） | [MATLAB R2022a 更新：Deep Network Designer](https://youtu.be/8nr25Gaz3Ss) |
| Classification Learner（R2022a 更新） | [MATLAB R2022a 更新：Classification Learner](https://youtu.be/Pe6CU9Jl0Kw) |
| Experiment Manager | [使用 Experiment Manager 管理深度學習實驗](https://youtu.be/NpJwoGspASg) |
| GPU 運算 | [在 MATLAB 中使用 GPU 加速運算](https://youtu.be/jPbflqw_aww) |

### 展示範例

| Demo | 影片 |
| --- | --- |
| GPU Coder | [GPU Coder 即時推論展示](https://youtu.be/MCWKbw5cf1c?si=KY5dmT7NoN1k-B4Q) |
| vSLAM | [vSLAM 視覺同步定位與建圖展示](https://youtu.be/bQuqNo13qPg?si=K4ZoYB6NDyJ-HOkJ) |
| DeepSORT | [DeepSORT 多目標追蹤展示](https://youtu.be/obj5VIQg5DU?si=8PiazwcDedwvR30n) |

[回到目錄](#目錄)

---

## 自製 App 與延伸專案

| 專案 | 說明 |
| --- | --- |
| [VoiceChat Bunny Robot](https://www.mathworks.com/matlabcentral/fileexchange/163996-voicechat-bunny-robot) | 語音互動與大型語言模型應用 |
| [ObjectDetectionAPP](https://github.com/MoonUsagi/ObjectDetectionAPP) | 以圖形化介面執行物件偵測 |
| [AOI_Layout](https://github.com/MoonUsagi/AIO_Layout) | 自動光學檢測介面與工作流程 |
| [Style_Transfer](https://github.com/MoonUsagi/Style_Transfer_APP) | 影像風格轉換 |
| [Image_Captioning](https://github.com/MoonUsagi/Image_Captioning_APP) | 影像描述生成 |
| [Image_Inpainting](https://github.com/MoonUsagi/Image_Inpainting) | 影像修補 |

[回到目錄](#目錄)

---

## 相關程式碼專案

| 領域 | 專案 |
| --- | --- |
| 影像處理與電腦視覺 | [IPCV_Lab](https://github.com/MoonUsagi/IPCV_Lab) |
| AOI 與視覺檢測 | [AOI_Lab](https://github.com/MoonUsagi/AOI_Lab/tree/main) |
| 影像分類 | [DL_Basic_Classificaiton](https://github.com/MoonUsagi/DL_Basic_Classificaiton) |
| 物件偵測與影像分割 | [DL_Advanced_RabbitDetect](https://github.com/MoonUsagi/DL_Advanced_RabbitDetect) |
| 深度學習延伸 | [DL_Exten](https://github.com/MoonUsagi/DL_Exten) |
| 數值深度學習 | [DL_Num](https://github.com/MoonUsagi/DL_Num) |
| 大型語言模型 | [llms-with-matlab](https://github.com/matlab-deep-learning/llms-with-matlab) |
| MATLAB 與 Python 整合 | [Python_MATLAB_Intergation](https://github.com/MoonUsagi/Python_MATLAB_Intergation) |
| Jetson Nano 部署 | [MATLAB_Mask_Detection-with-Jetoson-Nano](https://github.com/MoonUsagi/MATLAB_Mask_Detection-with-Jetoson-Nano) |

> [!TIP]
> 部分既有儲存庫名稱包含歷史拼字，例如 `Classificaiton`、`Intergation` 與 `Jetoson`。為確保連結可用，本頁保留原始儲存庫名稱。

## 未來規劃

- 補充 Vision Transformer（ViT）與新版遷移學習流程
- 增加 U-Net、SOLOv2、D-RISE 與影像異常偵測教學
- 建立強化學習（RL）學習路徑與 `RL_Lab`
- 增加 ARM 與 FPGA 部署案例
- 持續整理新版本 MATLAB 的 AI 功能更新

## 使用與貢獻

如果你發現失效連結、版本差異或希望新增主題，歡迎提出 Issue 或 Pull Request。新增資源時，建議同時標註：

- MATLAB 版本
- 所需 Toolbox
- 資源類型（影片、文章、程式碼或 App）
- 任務類別與建議先備知識

---

如果這份學習地圖對你有幫助，歡迎收藏並分享給其他 MATLAB 與 AI 開發者。
