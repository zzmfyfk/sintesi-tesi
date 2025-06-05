# Guida Completa alla Generazione di Immagini AI

## Introduzione: Come l'AI Crea le Immagini

Immagina di avere un artista digitale incredibilmente talennostroso che può dipingere qualsiasi cosa tu descriva a parole. Questo artista non usa pennelli e tele, ma algoritmi matematici che trasformano le tue descrizioni testuali in immagini meravigliose. Questa è l'essenza della generazione di immagini AI.

In questa guida, esploreremo quattro tecniche fondamentali:
- **Flux**: Il nuovo standard per la generazione di immagini di alta qualità
- **LoRA**: Come insegnare all'AI il nostro stile personale
- **ControlNet**: Come controllare precisamente la composizione delle immagini
- **Inpainting**: Come modificare e riparare parti specifiche delle immagini

---

## 1. Flux - La Generazione di Immagini di Nuova Generazione

### Che cos'è Flux?

Flux è come un pittore digitale super-intelligente con 12 miliardi di "neuroni artificiali" (parametri). Mentre i vecchi modelli erano come pittori che iniziavano con uno schizzo confuso e lo miglioravano gradualmente, Flux è come un artista che sa esattamente dove posizionare ogni pennellata fin dall'inizio.

**Analogia semplice**: Se Stable Diffusion tradizionale è come ripulire lentamente una foto sfocata, Flux è come tracciare una linea diretta dal punto A (il nostro prompt) al punto B (l'immagine perfetta).

### Come funziona Flux

```python
# Esempio base di Flux - semplificato per principianti
import torch
from diffusers import FluxPipeline

# Caricamento del modello - come aprire il programma di disegno
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16  # Formato numerico per risparmiare memoria
)

# Risparmia memoria del computer - come chiudere altri programmi
pipe.enable_model_cpu_offload()

# Descrivere cosa vogliamo - il nostro "prompt"
prompt = "Un gatto che tiene un cartello con scritto 'Ciao Mondo'"

# Generare l'immagine - come dire al pittore di iniziare
immagine = pipe(
    prompt,                    # Cosa disegnare
    height=1024,              # Altezza in pixel
    width=1024,               # Larghezza in pixel  
    guidance_scale=3.5,       # Quanto seguire alla lettera il prompt
    num_inference_steps=50,   # Quanti "passaggi" di pittura fare
    generator=torch.Generator("cpu").manual_seed(0)  # Numero per la riproducibilità
).images[0]

# Salvare il risultato
immagine.save("il_mio_gatto.png")
```

### Parametri Spiegati con Analogie

**Guidance Scale (3.5)** - *La precisione del pittore*
- Come dire a un pittore quanto deve essere preciso nel seguire le tue istruzioni
- **Valore basso (1-3)**: "Interpretalo liberamente, sii creativo"
- **Valore medio (3.5-7)**: "Segui le mie indicazioni ma usa la tua creatività"
- **Valore alto (10-15)**: "Fai esattamente quello che dico, senza inventare"

**Inference Steps (50)** - *Il tempo di lavorazione*
- Quante volte il pittore rivede e migliora il suo lavoro
- **Pochi passi (20)**: Schizzo veloce ma meno dettagliato
- **Molti passi (50)**: Lavoro accurato e dettagliato
- **Troppi passi (100+)**: Tempo sprecato senza miglioramenti evidenti

**Dimensioni (1024x1024)** - *La tela del pittore*
- Come la dimensione della tela su cui dipingere
- **Quadrata (1024x1024)**: Perfetta per ritratti e oggetti centrati
- **Orizzontale (1024x768)**: Ideale per paesaggi
- **Verticale (768x1024)**: Ottima per ritratti a figura intera

```python
# Esempio Schnell per generazione rapida
pipe_veloce = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", 
    torch_dtype=torch.bfloat16
)

# Configurazione per velocità massima
immagine_veloce = pipe_veloce(
    "Un tramonto in montagna",
    guidance_scale=0.0,        # Schnell non usa guidance
    num_inference_steps=4,     # Solo 4 passaggi
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
```

---

## 2. LoRA - Insegnare il nostro Stile all'AI

### Che cos'è LoRA?

LoRA (Low-Rank Adaptation) è come insegnare a un pittore già esperto il nostro stile personale senza farlo ricominciare da zero. Invece di insegnargli tutto da capo, gli mostri solo le "correzioni" e gli "aggiustamenti" per fare le cose a modo nostro.

**Analogia della scuola**: Se l'AI è uno studente che sa già dipingere bene, LoRA è come dargli un corso di specializzazione. Non cambiamo tutto quello che sa, ma aggiungiamo nuove competenze specifiche.

### Come Funziona LoRA

Immagina che il cervello dell'AI sia composto da migliaia di "interruttori" (parametri). Invece di cambiare tutti gli interruttori (che richiederebbe troppo tempo e risorse), LoRA aggiunge piccoli "regolatori di intensità" solo su alcuni interruttori specifici.

```python
# Esempio di training LoRA semplificato
from diffusers import StableDiffusionPipeline
from peft import LoraConfig
import torch

# Configurazione LoRA - le "impostazioni del corso di specializzazione"
configurazione_lora = LoraConfig(
    r=16,                  # Rank: quanti "regolatori" aggiungere
    lora_alpha=16,         # Alpha: intensità dei regolatori  
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Dove intervenire
    lora_dropout=0.1,      # Dropout: prevenire la "memorizzazione"
)

# Caricare il modello base - il nostro "studente esperto"
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Applicare LoRA - iniziare il "corso di specializzazione"
from peft import get_peft_model
pipeline.unet = get_peft_model(pipeline.unet, configurazione_lora)

# Ora il modello può imparare il nostro stile specifico
```

### Parametri LoRA Spiegati Semplicemente

**Rank (r=16)** - *La complessità dell'apprendimento*
- Come il numero di "lezioni speciali" che diamo al nostro pittore AI
- **Rank basso (4-8)**: Poche lezioni semplici, cambiano poco lo stile
- **Rank medio (16-32)**: Giusto numero di lezioni, buon equilibrio
- **Rank alto (64-128)**: Molte lezioni dettagliate, per stili complessi

**Alpha (16)** - *L'intensità dell'insegnamento*
- Quanto "forte" il nuovo stile dovrebbe influenzare l'AI
- **Formula semplice**: Solitamente Alpha = Rank o Alpha = 2×Rank
- **Alpha basso**: Il nuovo stile è sottile, si mescola delicatamente
- **Alpha alto**: Il nuovo stile è predominante e marcato

**Learning Rate (1e-4)** - *La velocità di apprendimento*
- Quanto velocemente l'AI dovrebbe imparare il nuovo stile
- **Analogia dell'auto**: Velocità di apprendimento come velocità di guida
- **Troppo veloce (1e-3)**: L'AI "frena" e non impara bene (overfitting)
- **Troppo lento (1e-6)**: L'AI impiega troppo tempo a imparare
- **Giusto (1e-4)**: Velocità ottimale per la maggior parte dei casi

**Batch Size (1-4)** - *Quante foto mostrare insieme*
- Come il numero di esempi che mostri al pittore in una volta
- **Batch Size 1**: Una foto alla volta (più lento ma funziona sempre)
- **Batch Size 4**: Quattro foto insieme (più veloce se hai memoria sufficiente)

### Preparare il Dataset per LoRA

```python
# Struttura delle cartelle per il training
# mio_progetto/
# ├── immagini/
# │   ├── 10_il_mio_personaggio/    # "10" = numero di ripetizioni
# │   │   ├── foto1.jpg
# │   │   ├── foto2.jpg
# │   │   └── ...
# │   └── metadata.csv             # Descrizioni delle foto

# Esempio di preparazione automatica del dataset
import os
from PIL import Image

def prepara_dataset(cartella_immagini, nome_soggetto, dimensione=512):
    """
    Prepara le immagini per il training LoRA
    
    cartella_immagini: dove sono le tue foto
    nome_soggetto: nome del personaggio/stile che vuoi insegnare
    dimensione: dimensione delle immagini (512 per SD1.5, 1024 per SDXL)
    """
    
    # Creare la struttura delle cartelle
    cartella_training = f"10_{nome_soggetto}"
    os.makedirs(cartella_training, exist_ok=True)
    
    for nome_file in os.listdir(cartella_immagini):
        if nome_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Aprire e ridimensionare l'immagine
            immagine = Image.open(os.path.join(cartella_immagini, nome_file))
            immagine = immagine.resize((dimensione, dimensione))
            
            # Salvare nella cartella di training
            nuovo_nome = f"{nome_soggetto}_{nome_file}"
            immagine.save(os.path.join(cartella_training, nuovo_nome))
    
    print(f"Dataset preparato in {cartella_training}")
```

### Training LoRA Step-by-Step

```python
# Comando di training semplificato (da eseguire nel terminale)
"""
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --dataset_name="il_mio_dataset" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --output_dir="mio_lora" \
  --rank=16
"""

# Spiegazione parametri di training:
# resolution=512: Dimensione immagini (512 per SD1.5)
# train_batch_size=1: Una immagine alla volta
# gradient_accumulation_steps=4: "Accumula" informazioni per 4 immagini
# max_train_steps=1500: Numero massimo di passaggi di apprendimento
# learning_rate=1e-04: Velocità di apprendimento (0.0001)
# lr_scheduler="cosine": Come diminuire la velocità nel tempo
```

### Usare un LoRA Addestrato

```python
# Usare il LoRA che hai creato
from diffusers import AutoPipelineForText2Image
import torch

# Caricare il modello base
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Caricare il nostro LoRA personalizzato
pipeline.load_lora_weights("percorso/al/mio_lora", weight_name="pytorch_lora_weights.safetensors")

# Generare immagini con il nostro stile
immagine = pipeline(
    "ritratto di [nome_trigger] che indossa un vestito blu",  # [nome_trigger] = parola speciale del nostro LoRA
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

immagine.save("risultato_con_mio_stile.png")
```

---

## 3. ControlNet - Il Controllo Preciso della Composizione

### Che cos'è ControlNet?

ControlNet è come dare a un pittore un "calco" o una "sagoma" da seguire. Invece di descrivere solo a parole cosa vuoi, gli mostri anche la forma, la posa, o la struttura che deve rispettare.

**Analogia del ricalco**: È come quando da bambini mettevamo un foglio sopra un disegno e lo ricalcavamo. ControlNet fa la stessa cosa: prende la "forma" di un'immagine guida e ci applica sopra lo stile che vuoi.

### Tipi di ControlNet

**Canny (Bordi)** - *Il contorno del disegno*
- Estrae solo i bordi e le linee principali di un'immagine
- Come un disegno a matita che mostra solo i contorni
- Perfetto per mantenere la composizione cambiando tutto il resto

```python
# Esempio Canny ControlNet
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Caricare ControlNet per i bordi
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Creare la pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Caricare un'immagine di riferimento
immagine_riferimento = Image.open("mia_foto.jpg")
immagine_array = np.array(immagine_riferimento)

# Estrarre i bordi - come fare un disegno a matita
soglia_bassa = 100      # Bordi meno evidenti
soglia_alta = 200       # Bordi più marcati
bordi = cv2.Canny(immagine_array, soglia_bassa, soglia_alta)

# Convertire in formato utilizzabile
bordi = bordi[:, :, None]
bordi = np.concatenate([bordi, bordi, bordi], axis=2)
immagine_controllo = Image.fromarray(bordi)

# Generare seguendo i bordi
risultato = pipe(
    "un castello medievale su una collina",
    image=immagine_controllo,           # L'immagine "guida"
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0   # Quanto seguire la guida (0-2)
).images[0]
```

**OpenPose (Pose Umane)** - *La posizione del corpo*
- Riconosce e copia la posa di una persona
- Come un manichino di legno che assumo la posizione che vuoi
- Ideale per ritratti e figure umane

```python
# Esempio OpenPose
from controlnet_aux import OpenposeDetector

# Rilevatore di pose - come un "sensore di movimento"
rilevatore_pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

# Estrarre la posa da una foto
foto_persona = Image.open("persona_in_posa.jpg")
mappa_pose = rilevatore_pose(foto_persona)

# Usare la posa per generare una nuova immagine
controlnet_pose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)

pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet_pose,
    torch_dtype=torch.float16
)

# Generare un robot nella stessa posa
robot_in_posa = pipe_pose(
    "un robot futuristico in una città cyberpunk",
    image=mappa_pose,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]
```

**Depth (Profondità)** - *La tridimensionalità*
- Capisce quanto sono vicini o lontani gli oggetti
- Come una mappa che mostra le montagne (vicino) e le valli (lontano)
- Perfetto per mantenere la spazialità 3D

**Scribble (Schizzo)** - *Il disegno a mano libera*
- Trasforma i nostroi schizzi in immagini dettagliate
- Come avere un assistente che "pulisce" e completa i nostroi disegni

### Parametri di Controllo

**ControlNet Conditioning Scale (0.0-2.0)** - *L'obbedienza alla guida*
- Quanto rigidamente seguire l'immagine di controllo
- **0.0**: Ignora completamente la guida
- **1.0**: Equilibrio perfetto tra guida e creatività
- **2.0**: Segue la guida molto rigidamente

**Starting/Ending Control Step** - *Quando applicare il controllo*
- Puoi decidere quando iniziare e smettere di seguire la guida
- **Start 0.0, End 1.0**: Controllo dall'inizio alla fine
- **Start 0.0, End 0.8**: Controllo iniziale, libertà creativa alla fine
- **Start 0.2, End 1.0**: Libertà iniziale, controllo preciso alla fine

### Combinare Più ControlNet

```python
# Usare più controlli contemporaneamente
from diffusers import MultiControlNetModel

# Caricare diversi tipi di controllo
controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")

# Combinarli
multi_controlnet = MultiControlNetModel([controlnet_canny, controlnet_pose])

# Usare entrambi i controlli
pipeline_multi = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=multi_controlnet,
    torch_dtype=torch.float16
)

# Generare con controlli multipli
risultato_complesso = pipeline_multi(
    "un supereroe in azione",
    image=[immagine_bordi, immagine_pose],        # Due immagini guida
    controlnet_conditioning_scale=[1.0, 0.8],    # Pesi diversi per ogni controllo
    num_inference_steps=20
).images[0]
```

---

## 4. Inpainting - Modificare e Riparare le Immagini

### Che cos'è l'Inpainting?

L'inpainting è come avere una "gomma magica" che può cancellare parti di un'immagine e riempirle intelligentemente con qualcosa di nuovo. È come quando un restauratore ripara un quadro antico, ma lo fa l'AI in pochi secondi.

**Analogia del puzzle**: Immagina di avere un puzzle completo, togliere alcuni pezzi, e chiedere a un artista molto bravo di disegnare pezzi nuovi che si incastrino perfettamente con quelli rimasti.

### Come Funziona l'Inpainting

L'inpainting usa una "maschera" (un'immagine in bianco e nero) per dire all'AI cosa cambiare:
- **Bianco**: "Cancella questo e disegna qualcosa di nuovo"
- **Nero**: "Lascia tutto com'è"
- **Grigio**: "Fai modifiche graduali"

```python
# Esempio base di inpainting
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# Caricare il modello per inpainting
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Caricare l'immagine originale e la maschera
immagine_originale = Image.open("mia_foto.png").resize((512, 512))
maschera = Image.open("maschera.png").resize((512, 512))

# Descrivere cosa mettere nella zona coperta dalla maschera
prompt = "un gatto giallo seduto su una panchina del parco"
prompt_negativo = "sfocato, bassa qualità, distorto"

# Fare l'inpainting
risultato = pipe_inpaint(
    prompt=prompt,
    negative_prompt=prompt_negativo,
    image=immagine_originale,      # Immagine da modificare
    mask_image=maschera,           # Cosa modificare (bianco = cambia)
    num_inference_steps=50,        # Quanti passaggi di "pittura"
    guidance_scale=7.5,            # Quanto seguire il prompt
    strength=0.75                  # Quanto cambiare (0.0-1.0)
).images[0]

risultato.save("immagine_modificata.png")
```

### Parametri dell'Inpainting Spiegati

**Strength (0.0-1.0)** - *L'intensità della modifica*
- Quanto l'AI dovrebbe cambiare la zona mascherata
- **0.0**: Non cambia nulla (inutile)
- **0.3-0.5**: Modifiche sottili, buono per ritocchi viso
- **0.75**: Valore standard per la maggior parte dei casi
- **1.0**: Cambiamento completo, potrebbe perdere coerenza

**Guidance Scale** - *La fedeltà al prompt*
- Stesso significato del normale Stable Diffusion
- **7.5**: Valore equilibrato raccomandato
- **Più alto**: Segue più precisamente la descrizione
- **Più basso**: Più creatività artistica

**Inference Steps** - *La qualità del lavoro*
- **20-25**: Veloce, qualità buona
- **50**: Qualità alta standard
- **100+**: Non migliora molto, spreca tempo

### Creare Maschere Automaticamente

```python
# Creare maschere con codice invece che a mano
from PIL import Image, ImageDraw
import numpy as np

def crea_maschera_cerchio(dimensioni, centro, raggio):
    """
    Crea una maschera circolare
    
    dimensioni: (larghezza, altezza) dell'immagine
    centro: (x, y) del centro del cerchio
    raggio: dimensione del cerchio
    """
    
    maschera = Image.new('L', dimensioni, 0)  # Immagine nera
    disegno = ImageDraw.Draw(maschera)
    
    # Disegnare un cerchio bianco
    x, y = centro
    disegno.ellipse([x-raggio, y-raggio, x+raggio, y+raggio], fill=255)
    
    return maschera

# Esempio: rimuovere un oggetto al centro dell'immagine
maschera_cerchio = crea_maschera_cerchio((512, 512), (256, 256), 50)

def crea_maschera_rettangolo(dimensioni, angolo_sup_sin, angolo_inf_des):
    """
    Crea una maschera rettangolare
    
    dimensioni: (larghezza, altezza)
    angolo_sup_sin: (x, y) dell'angolo superiore sinistro
    angolo_inf_des: (x, y) dell'angolo inferiore destro
    """
    
    maschera = Image.new('L', dimensioni, 0)
    disegno = ImageDraw.Draw(maschera)
    
    # Disegnare un rettangolo bianco
    disegno.rectangle([angolo_sup_sin, angolo_inf_des], fill=255)
    
    return maschera

# Esempio: cambiare lo sfondo (rettangolo che copre tutto tranne il centro)
maschera_sfondo = crea_maschera_rettangolo((512, 512), (0, 0), (512, 200))
```

### Applicazioni Pratiche dell'Inpainting

**Rimozione di Oggetti**
```python
# Rimuovere una persona da una foto di gruppo
prompt_rimozione = "prato verde, paesaggio naturale"
risultato_pulizia = pipe_inpaint(
    prompt=prompt_rimozione,
    image=foto_gruppo,
    mask_image=maschera_persona,
    strength=0.8,  # Cambiamento forte per rimozione completa
    guidance_scale=8.0
).images[0]
```

**Cambio di Sfondi**
```python
# Cambiare lo sfondo di un ritratto
prompt_nuovo_sfondo = "biblioteca antica con libri, illuminazione soffusa"
ritratto_nuovo_sfondo = pipe_inpaint(
    prompt=prompt_nuovo_sfondo,
    image=ritratto_originale,
    mask_image=maschera_sfondo,
    strength=0.9,
    guidance_scale=7.5
).images[0]
```

**Ritocchi e Miglioramenti**
```python
# Migliorare dettagli del viso
prompt_ritocco = "pelle liscia e naturale, illuminazione morbida"
viso_migliorato = pipe_inpaint(
    prompt=prompt_ritocco,
    image=ritratto,
    mask_image=maschera_pelle,
    strength=0.3,  # Cambiamento sottile per naturalezza
    guidance_scale=6.0
).images[0]
```

### Tecniche Avanzate di Inpainting

**Inpainting con Modelli SDXL** (Alta Risoluzione)
```python
# Per immagini ad alta risoluzione
from diffusers import AutoPipelineForInpainting

pipe_sdxl = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16
).to("cuda")

# Processare immagini 1024x1024
immagine_hd = Image.open("foto_ad_alta_risoluzione.jpg").resize((1024, 1024))
maschera_hd = Image.open("maschera_hd.png").resize((1024, 1024))

risultato_hd = pipe_sdxl(
    prompt="paesaggio montano dettagliato",
    image=immagine_hd,
    mask_image=maschera_hd,
    guidance_scale=8.0,
    num_inference_steps=20,
    strength=0.85
).images[0]
```

---

## Consigli Pratici per Principianti

### Iniziare con Progetti Semplici

**Primo Progetto: Generazione Base con Flux**
1. Installa Flux seguendo le istruzioni
2. Inizia con prompt semplici: "un gatto rosso su un prato"
3. Sperimenta con i parametri base
4. Salva i risultati che ti piacciono

**Secondo Progetto: Controllare la Composizione**
1. Trova una foto con una composizione che ti piace
2. Usa Canny ControlNet per estrarre i bordi
3. Genera nuove immagini seguendo quella struttura
4. Prova diversi prompt mantenendo la stessa composizione

**Terzo Progetto: Personalizzare con LoRA**
1. Raccogli 20-30 foto di un soggetto specifico
2. Addestra un LoRA semplice
3. Genera immagini del nostro soggetto in situazioni diverse
4. Documenta cosa funziona meglio

### Errori Comuni da Evitare

**Errore 1: Prompt troppo complicati all'inizio**
- ❌ "Un guerriero elfico con armatura dorata che cavalca un drago blu in una tempesta di fulmini durante un eclissi lunare"
- ✅ "Un guerriero elfico con armatura dorata"

**Errore 2: Non controllare le dimensioni delle immagini**
- Usa sempre le dimensioni giuste per ogni modello
- SD 1.5: 512x512
- SDXL: 1024x1024
- Flux: 1024x1024

**Errore 3: Ignorare i prompt negativi**
- Usa sempre prompt negativi per migliorare la qualità
- Esempi base: "sfocato, bassa qualità, distorto, artefatti"

### Configurazione Hardware Raccomandate

**Setup Minimo**
- GPU: RTX 3070 (8GB VRAM)
- RAM: 16GB
- Spazio: 50GB per i modelli

**Setup Raccomandato**
- GPU: RTX 4080/4090 (12-24GB VRAM)
- RAM: 32GB
- Spazio: 100GB+ su SSD

**Setup Budget**
- Usa servizi cloud come Google Colab
- Installa versioni ottimizzate per GPU più piccole
- Usa modelli quantizzati per risparmiare memoria

### Organizzare il nostro Workflow

```python
# Crea uno script organizzato per i nostroi progetti
import os
from datetime import datetime

def organizza_progetto(nome_progetto):
    """Crea la struttura delle cartelle per un nuovo progetto"""
    
    data_oggi = datetime.now().strftime("%Y%m%d")
    cartella_progetto = f"{data_oggi}_{nome_progetto}"
    
    # Creare le cartelle
    os.makedirs(f"{cartella_progetto}/input", exist_ok=True)
    os.makedirs(f"{cartella_progetto}/output", exist_ok=True)
    os.makedirs(f"{cartella_progetto}/maschere", exist_ok=True)
    os.makedirs(f"{cartella_progetto}/esperimenti", exist_ok=True)
    
    # Creare un file per annotare le impostazioni
    with open(f"{cartella_progetto}/settings.txt", "w") as f:
        f.write(f"Progetto: {nome_progetto}\n")
        f.write(f"Data: {data_oggi}\n")
        f.write("Impostazioni utilizzate:\n")
        f.write("- Modello: \n")
        f.write("- Parametri: \n")
        f.write("- Note: \n")
    
    print(f"Progetto '{nome_progetto}' creato in {cartella_progetto}")
    return cartella_progetto

# Esempio d'uso
organizza_progetto("ritratti_fantasy")
```

---

