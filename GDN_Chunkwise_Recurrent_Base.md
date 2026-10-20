# GDN (Gated Delta Networks) 適合硬體 Chunk-wise 並行計算的遞迴基底

對於 **GDN (Gated Delta Networks)** 而言，要在現代 GPU 上進行高效的 **Chunk-wise（分塊）並行訓練**，其核心難點在於公式中同時存在**數據驅動的衰減標量 $\alpha_t$** 與基於 **Delta Rule 的 Rank-1 修正項**。

如果直接按 Token 逐步遞推，在 GPU 上會退化為極慢的序列串行運算。為了實現並行，GDN 擴展了 DeltaNet 的 **延伸 WY 表示法（Extended WY Representation）**，將跨 Token 的複雜動態遞迴，轉化為**塊內完全並行矩陣運算（Chunk-internal parallel）**與**塊間狀態序列傳遞（Chunk-level cross-state passing）**。

---

## 一、 核心挑戰與代數重構（Householder 形式）

將 GDN 的步進更新公式展開並以舊狀態 $S_{t-1}$ 歸項：
$$S_t = \alpha_t S_{t-1} (I - \beta_t k_t k_t^\top) + \beta_t k_t v_t^\top$$

這在本質上是一個**廣義豪斯霍爾德變換（Generalized Householder Transformation）**配上新資訊寫入。當我們將序列切分為大小為 $C$ 的分塊（Chunk）時，若已知該塊起始的歷史狀態為 $S_{\text{init}}$，則該塊內任意位置 $t$（$1 \le t \le C$）的狀態可被表示為：
$$S_t = A_{1:t} S_{\text{init}} + \sum_{i=1}^t B_{i:t} k_i v_i^\top$$
*其中 $A_{1:t}$ 和 $B_{i:t}$ 是由 $\alpha$ 與 $\beta k k^\top$ 複合而成的累積消去矩陣。*

---

## 二、 適合硬體的擴展 WY 表示法（Extended WY Representation）

為了避免在晶片上（SRAM / Triton 核心）顯式地去計算與維存複雜的 $d_k \times d_k$ 累積消去矩陣，GDN 透過數學歸納法，將整個 Chunk 內的所有歷史消去矩陣與權重歸併到兩個累積緩衝矩陣：**$W \in \mathbb{R}^{C \times d_k}$** 與 **$U \in \mathbb{R}^{C \times d_k}$**。

這使得一個 Chunk 結束後的終態 $S_{\text{end}}$（即傳遞給下一個 Chunk 的遞迴基底）可以被完美寫成一組**純矩陣乘法**：

### 1. 塊間傳遞的遞迴基底 (Chunk-level Recurrent Base)
$$S_{\text{end}} = \Lambda S_{\text{init}} + U^\top V$$
* **$\Lambda$**：該 Chunk 內全體遺忘門的總乘積（是一個對角矩陣或純量縮放）。
* **$U^\top V$**：將整個塊內的輸入 $V \in \mathbb{R}^{C \times d_v}$ 透過並行構建的矩陣 $U$ 一次性寫入狀態。

### 2. 核心：塊內變數 $U$ 的構造形式（上三角反演）
在硬體中，矩陣 $U$ 的每一行 $u_t^\top$ 包含了過往所有步驟的 Delta 糾正。GDN 將其重構為一個**上三角矩陣（Upper Triangular, UT）的反演問題**，在 GPU 內實現 $\mathcal{O}(d)$ 複雜度並行求值：
$$U = (I - L_{\alpha, \beta})^{-1} W$$

在 Triton 算子實現中，具體的塊內構造基底如下：
1. **構造基礎項 $W$**：$W = \text{Diag}(\boldsymbol{\beta}) K$，即直接對輸入的 Key 向量乘以更新率。
2. **構造塊內干擾矩陣 $L_{\alpha, eta} \in \mathbb{R}^{C \times C}$**：
   其矩陣中位置 $(t, i)$ 的元素（其中 $t > i$）定義為：
   $$[L_{\alpha, \beta}]_{t, i} = \beta_t \left( \prod_{j=i+1}^t \alpha_j \right) (k_t^\top k_i)$$
   * **物理意義**：這代表了第 $i$ 步寫入的 Key 與第 $t$ 步輸入的 Key 之間的衰減關聯度。
3. **並行求解 $U$**：由於 $L_{\alpha, \beta}$ 是一個嚴格下三角矩陣，GPU 可以使用極快的 `trsm`（上/下三角矩陣方程求解器）直接計算出 $U = (I - L_{\alpha, \beta})^{-1} W$。

---

## 三、 完整 Chunk-wise 訓練演算法流程

理解了上述遞迴基底後，GDN 在硬體（如 Triton）中的前向並行計算被拆解為以下三個高並行化的高效步驟：

```
⚡ 步驟 1：[全序列完全並行] (SRAM 塊內並行)
   對所有 Chunk 同時計算各自塊內的跨 Token 干擾矩陣 L，
   並透過上三角反演求出每個塊專屬的累積矩陣 W 與 U。
      │
      ▼
🔄 步驟 2：[跨跨塊串行傳遞] (序列表現為大 RNN 基底)
   For chunk = 1 to Num_Chunks:
       1. 讀取上一個塊留下的狀態 S_init
       2. 計算當前塊內被 Delta Rule 修正後的新 Value 矩陣：V_new = V - U @ S_init.T
       3. 根據遞迴基底更新並儲存下一個塊的起點狀態：S_end = Λ * S_init + U.T @ V_new
      │
      ▼
⚡ 步驟 3：[全序列完全並行] (輸出並行計算)
   利用各塊算好的狀態矩陣，直接結合 Query 矩陣並行點積輸出最終結果：
   Output = Q @ S_init.T + (並行塊內殘差 Attention)
```

透過這種將「$d_k \times d_k$ 的矩陣更新」轉移為「$C \times C$ 上三角矩陣反演（通常分塊大小 $C=64$ 或 $128$）」的代數變換，GDN 成功在硬體層面實現了與序列長度 $L$ 呈**線性複雜度 $\mathcal{O}(L)$** 的極致並行加速。