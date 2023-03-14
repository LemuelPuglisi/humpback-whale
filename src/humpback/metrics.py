import numpy as np
import faiss

from humpback.utils import images_to_embeddings


def map_per_image(label, predictions):
    try:
        return 1 / (predictions[:5].tolist().index(label) + 1)    
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])


def evaluate_model(encoder, testset, device='cuda', embedding_dim=512):        
    print('Extracting Whales ground truth IDs.')
    whales = testset.annotations.label.to_numpy()
    
    print('Generating the embeddings.')
    encoder.eval()
    encoder = encoder.to(device)
    
    embeddings = images_to_embeddings(encoder, testset, device=device).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print('Generating FAISS index.')
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(embeddings) # type: ignore
    
    print('Retrieving similar images.')
    _, results = faiss_index.search(embeddings, 6) # type: ignore
    predicted_whales = whales[ results[:, 1:6] ]

    map5 = map_per_set(whales, predicted_whales)
    return map5


def evaluate_model_nw(encoder, testset, device='cuda', embedding_dim=512, threshold=0.55):   
    print('Extracting Whales ground truth IDs.')
    whales = testset.annotations.label.to_numpy()
    
    print('Generating the embeddings.')
    encoder.eval()
    encoder = encoder.to(device)
    
    embeddings = images_to_embeddings(encoder, testset, device=device).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print('Generating FAISS index.')
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(embeddings) # type: ignore
    
    print('Retrieving similar images.')
    similarities, results = faiss_index.search(embeddings, 6) # type: ignore
    similarities = similarities[:, 1:6]
    predicted_whales = whales[ results[:, 1:6] ]


    for res, sim in zip(predicted_whales, similarities):
        for i, s in enumerate(sim):
            if s < threshold:
                tmp = res[i]
                res[i] = -1
                if (i < 4): res[i+1] = tmp
                break

    map5 = map_per_set(whales, predicted_whales)
    return map5