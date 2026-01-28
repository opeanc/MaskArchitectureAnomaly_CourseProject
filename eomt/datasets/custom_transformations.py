import cv2
import numpy as np
import pandas as pd
import random


def load_sample(df_annotations):


    # random sample an anomaly object
    row = df_annotations.sample(1).iloc[0]
    label_type = row['label'] # "air", "ground", o "obj"

    return (row['image_path'], row['mask_path'], label_type)



def extract_object(image_path, mask_path):
    """
    Ritaglia l'oggetto dall'immagine originale usando la maschera binaria.
    Restituisce l'immagine ritagliata e la sua maschera ritagliata.
    """
    # Load image and mask
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarization    
    # To make sure mask is perfectly black (0) or white (255)
    # all the values > 127 will become 255, values <= 127 will be transformed in 0
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Bounding Box
    # Questo serve a ritagliare il rettangolo minimo che contiene il cane,
    # eliminando tutto lo spazio nero inutile intorno.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Nessun oggetto trovato nella maschera!")
        return None, None

    # Prendi il contorno più grande (nel caso ci siano piccoli artefatti)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 4. Ritaglia (Crop)
    # Ritagliamo sia l'immagine a colori che la maschera usando le coordinate
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = binary_mask[y:y+h, x:x+w]

    # 5. Mascheratura Sfondo (Opzionale ma pulito)
    # Applichiamo la maschera all'immagine ritagliata per rendere nero
    # lo sfondo immediato (l'erba che potrebbe essere rimasta dentro il rettangolo)
    # Questo passaggio è fondamentale per non incollare pezzetti di erba.
    cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

    return cropped_img, cropped_mask


def get_random_scale(img, mask, min_scale=0.3, max_scale=1.0):
    """
    Ridimensiona l'oggetto casualmente, mantenendo l'aspect ratio.
    """
    h, w = img.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Evita dimensioni 0
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    new_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return (new_image, new_mask)


def check_ego_vehicle_overlap(bg_mask, x, y, h_obj, w_obj, void_id=255, threshold=0.3):
    """
    Controlla se la posizione proposta si sovrappone all'ego-vehicle.
    In Cityscapes, l'ego-vehicle è sempre etichettato come VOID (255).
    Inoltre, l'ego-vehicle è sempre nella parte bassa dell'immagine.
    """
    H, W = bg_mask.shape

    # Estrai la ROI dalla maschera di background
    roi = bg_mask[y:y+h_obj, x:x+w_obj]

    # Calcoliamo quanti pixel nella ROI sono 'Void' (255)
    void_pixels = np.sum(roi == void_id)
    total_pixels = h_obj * w_obj
    void_ratio = void_pixels / total_pixels

    # Se l'oggetto è troppo sovrapposto al Void (spesso bordi neri o ego-vehicle), scartiamo.
    # Aggiungiamo un check posizionale: l'ego vehicle è in basso al centro.
    # Se y è molto basso (es. > 80% dell'immagine) e c'è void, è quasi sicuramente l'auto.
    is_bottom = y > (H * 0.75)

    if is_bottom and void_ratio > 0.1: # Tolleranza stretta in basso
        return True # Sovrapposizione rilevata (Rifiuta)

    if void_ratio > threshold: # Tolleranza più ampia altrove (es. bordi neri)
        return True

    return False # Posizione valida

def get_smart_position(bg_shape, obj_shape, label_type, bg_mask, max_attempts=50, margin=70):
    """
    Trova una coordinata (x, y) valida con un margine di sicurezza dai bordi.
    margin: distanza minima in pixel dai bordi dell'immagine.
    """
    H_bg, W_bg = bg_shape[:2]
    h_obj, w_obj = obj_shape[:2]

    # Calcoliamo i limiti effettivi dove possiamo mettere l'angolo top-left (x, y)
    # L'oggetto deve finire a (W - margin), quindi x non può superare (W - w - margin)
    # Inoltre x non può essere minore di margin.

    x_min = margin
    x_max = W_bg - w_obj - margin

    y_min = margin
    y_max = H_bg - h_obj - margin

    # Se l'oggetto è troppo grande per rispettare i margini, riduciamo le pretese
    if x_max <= x_min or y_max <= y_min:
        # Fallback: prova a inserirlo senza margini (o rifiuta se proprio non entra)
        x_min, y_min = 0, 0
        x_max = W_bg - w_obj
        y_max = H_bg - h_obj
        if x_max < 0 or y_max < 0:
            return None # L'oggetto è più grande dello sfondo!

    for _ in range(max_attempts):
        # 1. Determina la Y (Altezza)
        # Nota: y_max è già calcolato considerando il margine inferiore

        if label_type == 'air':
            # Parte alta
            limit = max(y_min, y_max // 2)
            y = random.randint(y_min, limit)

        elif label_type == 'ground':
            # Parte bassa
            # Evitiamo di partire troppo in alto, ma rispettiamo y_min
            start_y = max(y_min, y_max // 3)
            y = random.randint(start_y, y_max)

        else: # label_type == 'obj'
            y = random.randint(y_min, y_max)

        # 2. Determina la X (Larghezza)
        x = random.randint(x_min, x_max)

        # 3. Controllo Ego-Vehicle
        # Passiamo le coordinate candidate
        if not check_ego_vehicle_overlap(bg_mask, x, y, h_obj, w_obj):
            return (x, y)

    return None



def augment_object(img, mask):
    """
    Applica Data Augmentation geometrica e fotometrica all'oggetto COCO.
    Args:
        img: Immagine oggetto ritagliata (BGR)
        mask: Maschera oggetto (Gray)
    """
    h, w = img.shape[:2]

    # --- 1. GEOMETRIC: Random Flip ---
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    # --- 2. GEOMETRIC: Random Rotation (-15° a +15°) ---
    # Ruotare aiuta molto, ma non esagerare o l'oggetto viene tagliato
    if random.random() < 0.7: # 70% di probabilità
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)

        # Matrice di rotazione
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Applica rotazione
        # IMPORTANTE: borderValue=0 (nero) per evitare artefatti ai bordi
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        # IMPORTANTE: maschera usa INTER_NEAREST per restare binaria
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # --- 3. PHOTOMETRIC: Color Jitter (Brightness & Contrast) ---
    # Modifichiamo leggermente i colori PRIMA che il blending li uniformi.
    # Questo aggiunge "carattere" all'anomalia.
    if random.random() < 0.8:
        # Contrasto: moltiplicatore (es. 0.8 a 1.2)
        alpha = random.uniform(0.8, 1.2)
        # Luminosità: addendo (es. -30 a +30)
        beta = random.uniform(-30, 30)

        # Formula: pixel = alpha * pixel + beta
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- 4. PHOTOMETRIC: Saturation Jitter ---
    # A volte cambiamo la saturazione per avere oggetti più/meno vivi
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        # Moltiplica canale S (Saturazione)
        sat_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 1] *= sat_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    return img, mask


def perform_color_transfer(source, target, strength=0.4):
    """
    Applica il color transfer ma lo mixa con l'originale per evitare
    l'effetto "fantasma/camouflage".

    Args:
        strength (float): 0.0 = Colore Originale, 1.0 = Totale Camouflage.
                          Un valore tra 0.4 e 0.6 è ideale.
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, l_std_src) = cv2.meanStdDev(source_lab)
    (l_mean_tar, l_std_tar) = cv2.meanStdDev(target_lab)

    # Applica il transfer matematico completo
    source_lab_t = source_lab.copy()

    # Sottrai media source
    source_lab_t -= np.array([l_mean_src.flatten()]).astype("float32")

    # Scala deviazione standard
    # Limitiamo lo scaling: se lo sfondo è troppo piatto (std vicina a 0),
    # l'oggetto perde tutti i dettagli. Aggiungiamo un clamp.
    scale = l_std_tar.flatten() / (l_std_src.flatten() + 1e-5)
    scale = np.clip(scale, 0.5, 1.5) # Evita contrasti troppo esplosivi o piatti

    source_lab_t *= scale

    # Aggiungi media target
    source_lab_t += np.array([l_mean_tar.flatten()]).astype("float32")
    source_lab_t = np.clip(source_lab_t, 0, 255)

    # Converti il risultato trasferito in BGR
    transfer = cv2.cvtColor(source_lab_t.astype("uint8"), cv2.COLOR_LAB2BGR)

    # --- IL TRUCCO: BLENDING ---
    # Mixiamo l'immagine originale (source) con quella trasferita (transfer)
    # strength basso -> più simile all'originale
    # strength alto -> più simile allo sfondo
    final_result = cv2.addWeighted(transfer, strength, source, 1.0 - strength, 0)

    return final_result

def apply_depth_blur(img, y_pos, total_height):
    """
    Simula "depth blur based on the position".
    Assunzione: Più in alto è l'oggetto nell'immagine, più è lontano/sfocato.
    """
    # Normalizza la posizione Y tra 0 e 1
    rel_y = y_pos / total_height

    # Se l'oggetto è nella metà superiore (lontano) o molto in basso (troppo vicino/mosso)
    # applichiamo un blur.
    # Logica semplificata: Più in alto = più blur (sfocatura distanza)
    # Cityscapes horizon è circa a 0.4-0.5 dell'altezza.

    k_size = 0
    if rel_y < 0.5: # Lontano / Orizzonte
        k_size = 3
    elif rel_y > 0.85: # Vicinissimo (Motion Blur rapido della strada)
        k_size = 5

    if k_size > 0:
        return cv2.GaussianBlur(img, (k_size, k_size), 0)

    return img

def add_noise(img, strength=0.05):
    """
    Aggiunge "color noise"  per simulare la grana del sensore.
    """
    noise = np.random.randn(*img.shape).astype(np.float32)
    # Scaliamo il rumore e lo aggiungiamo
    noisy_img = img.astype(np.float32) + (noise * strength * 255)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)




def inject_anomaly(city_img, city_mask, coco_img_path, coco_mask_path, label, anomaly_id=254):
    """
    Funzione Principale:
    1. Estrae l'oggetto da COCO.
    2. Esegue il blending sull'immagine RGB (Smooth + Brightness).
    3. Aggiorna la maschera di segmentazione (Hard Paste).

    Args:
        city_img: immagine cityscapes
        city_mask: maschera cityscapes
        coco_img_path: PATH immagine oggetto
        coco_mask_path: PATH maschera oggetto
        label: label dell'oggetto
        anomaly_id: L'ID numerico da assegnare ai pixel dell'anomalia nella maschera (es. 254)
    """

    # 1. ESTRAZIONE AL VOLO
    obj_img, obj_mask = extract_object(coco_img_path, coco_mask_path)

    if obj_img is None:
        # Se qualcosa va storto nell'estrazione, ritorna gli originali senza modifiche
        print("Attenzione: Oggetto non trovato nella maschera COCO.")
        return city_img, city_mask

    # Data augmentation
    obj_img, obj_mask = augment_object(obj_img, obj_mask)

    # suggerito dal paper di fishyscapes
    obj_img_rescale, obj_mask_rescale = get_random_scale(obj_img, obj_mask, min_scale=0.3, max_scale=1.0)

    # Dimensioni dell'oggetto ritagliato
    h_obj, w_obj = obj_img_rescale.shape[:2]

    position = get_smart_position(city_img.shape, obj_img_rescale.shape, label, city_mask)
    print(position)

    if position is None:
        return city_img, city_mask

    x_pos, y_pos = position

    # Controlli di sicurezza sui bordi (per evitare crash se l'oggetto esce dall'immagine)
    if x_pos + w_obj > city_img.shape[1] or y_pos + h_obj > city_img.shape[0]:
        print("Attenzione: L'oggetto esce dai bordi dell'immagine Cityscapes.")
        return city_img, city_mask

    # --- FASE A: BLENDING AVANZATO (NUOVO) ---
    roi = city_img[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj]

    # 1. COLOR TRANSFER (Sostituisce Brightness Adaptation)
    # Trasferisce l'atmosfera "grigia" di Cityscapes sul cane "colorato"
    obj_img_adjusted = perform_color_transfer(obj_img_rescale, roi)

    # 2. DEPTH BLUR
    # Sfoca leggermente se l'oggetto è lontano
    obj_img_adjusted = apply_depth_blur(obj_img_adjusted, y_pos, city_img.shape[0])

    # 3. NOISE
    # Aggiunge grana per uniformare
    obj_img_adjusted = add_noise(obj_img_adjusted)

    # 4. ALPHA SMOOTHING (Invariato)
    float_mask = obj_mask_rescale.astype(float) / 255.0
    float_mask_blurred = cv2.GaussianBlur(float_mask, (3, 3), 0) # Kernel ridotto per dettagli fini
    alpha = np.dstack([float_mask_blurred] * 3)

    # 5. COMPOSITING
    blended_roi = (alpha * obj_img_adjusted) + ((1 - alpha) * roi)

    final_img = city_img.copy()
    final_img[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = blended_roi.astype(np.uint8)

    # --- FASE B: AGGIORNAMENTO MASCHERA (INVARIATO) ---
    final_mask = city_mask.copy()
    roi_mask = final_mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj]
    roi_mask[obj_mask_rescale > 127] = anomaly_id
    final_mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = roi_mask

    return final_img, final_mask
