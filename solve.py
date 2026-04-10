import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from transformers import AutoModel

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
BASE_DIR = os.path.dirname(__file__)

ALL_SAMPLES = os.path.join(BASE_DIR, "all_libraries_filtered_predictions.json")
EDAPI_SAMPLES = os.path.join(BASE_DIR, "deepseek_edapi.json")

THRESHOLD = 0.8
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def cosine_similarity(vec_a, vec_b):
#     return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def get_embeddings(data_list, model):
    prompts = [item["prompt"] for item in data_list]
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Encoding"):
            batch_prompts = prompts[i : i + BATCH_SIZE]
            batch_emb = model.encode(batch_prompts)
            batch_emb = torch.tensor(batch_emb, device=DEVICE)
            embeddings.append(batch_emb)
            
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def main():
    print(f"Using: {DEVICE}")

    # ==========================================
    # 1. ĐỌC VÀ LỌC EXACT MATCH (Khớp hoàn toàn)
    # ==========================================
    print("Đang đọc dữ liệu JSON...")
    with open(ALL_SAMPLES, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    with open(EDAPI_SAMPLES, 'r', encoding='utf-8') as f:
        edapi_data = json.load(f)

    edapi_prompts = set(item["prompt"] for item in edapi_data)
    filtered_exact_data = [item for item in all_data if item["prompt"] not in edapi_prompts]

    with open('filtered_exact_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_exact_data, f, ensure_ascii=False, indent=4)

    print(f"Số lượng mẫu ban đầu trong all_data: {len(all_data)}")
    print(f"Số lượng mẫu trong edapi_data: {len(edapi_data)}")
    print(f"Số lượng mẫu giữ lại sau khi lọc exact match: {len(filtered_exact_data)}")

    # ==========================================
    # 2. KHỞI TẠO MÔ HÌNH NHÚNG (EMBEDDING MODEL)
    # ==========================================
    print("\nKhởi tạo model Jina...")
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-code", 
        trust_remote_code=True
    )
    model = model.to(DEVICE)
    model.eval()

    # ==========================================
    # 3. TRÍCH XUẤT EMBEDDINGS
    # ==========================================
    print("\nTrích xuất embeddings cho edapi_data...")
    edapi_embs = get_embeddings(edapi_data, model)
    
    print("\nTrích xuất embeddings cho all_data...")
    all_embs = get_embeddings(all_data, model)

    # ==========================================
    # 4. LỌC BƯỚC 1: all_data vs edapi_data
    # ==========================================
    print("\nThực hiện Bước 1: Lọc all_data trùng với edapi_data...")
    
    similarity_matrix_step1 = torch.mm(all_embs, edapi_embs.t())
    max_sims_step1, _ = torch.max(similarity_matrix_step1, dim=1)

    valid_indices_step1 = (max_sims_step1 <= THRESHOLD).nonzero(as_tuple=True)[0]

    step1_filtered_data = [all_data[i] for i in valid_indices_step1.tolist()]
    step1_filtered_embs = all_embs[valid_indices_step1]

    print(f"-> Số mẫu còn lại sau Bước 1: {len(step1_filtered_data)}")

    # ==========================================
    # 5. LỌC BƯỚC 2: Loại bỏ trùng lặp nội bộ (Deduplication)
    # ==========================================
    print("\nThực hiện Bước 2: Loại bỏ trùng lặp trong nội bộ all_data (Greedy Filter)...")

    final_kept_data = []
    if len(step1_filtered_data) > 0:
        # Bắt đầu với mẫu đầu tiên
        final_kept_data.append(step1_filtered_data[0])
        kept_embs = step1_filtered_embs[0].unsqueeze(0)

        for i in tqdm(range(1, len(step1_filtered_data)), desc="Deduplicating"):
            curr_emb = step1_filtered_embs[i].unsqueeze(0)
            
            # Tính similarity của mẫu hiện tại với TẤT CẢ các mẫu ĐÃ ĐƯỢC GIỮ LẠI
            sims = torch.mm(curr_emb, kept_embs.t())
            
            # Nếu không có mẫu nào trong tập đã giữ có độ tương đồng > THRESHOLD, ta giữ mẫu này
            if sims.max() <= THRESHOLD:
                final_kept_data.append(step1_filtered_data[i])
                kept_embs = torch.cat((kept_embs, curr_emb), dim=0)

    print(f"-> Số mẫu còn lại sau Bước 2: {len(final_kept_data)}")

    # ==========================================
    # 6. LƯU KẾT QUẢ CUỐI CÙNG
    # ==========================================
    with open('filter_cosine_all.json', 'w', encoding='utf-8') as f:
        json.dump(final_kept_data, f, ensure_ascii=False, indent=4)
        
    print("\nHoàn tất lưu file 'filter_cosine_data.json'!")

if __name__ == "__main__":
    main()