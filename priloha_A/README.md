# **ScalesCounter** - tvorba modelů konvolučních neuronových sítí
Tento program byl vytvořen v rámci diplomové práce na Ústavu automatizace a informatiky Fakulty strojního inženýrství VUT v Brně. Zároveň slouží vědeckým účelům pro Akademii věd České republiky.

Autor: **Štěpán Maršala**

## Úvod
Sada přiložených skriptů slouží k:
- úpravě a štítkování datové sady,
- natrénovaní a evaluaci modelů sítí YOLOv4 a U-Net.

Modely konvolučních neuronových sítí slouží k běhu programu ScalesCounter, který počítá sekundární šupiny ještěrek obecných na jejich ventrálních stranách (břichách) a výsledky zapisuje do textového souboru. Dvě verze tohoto programu jsou součástí další přílohy.

## Konfigurace parametrů
Před spouštěním jednotlivých skriptů je nejprve potřeba nakonfigurovat proměnné v souboru **configure.py**. Soubor **configure.py** je editovatelný v běžných textových editorech jako je například poznámkový blok ve Windows. Výchozí hodnoty jsou vyzkoušené a uživatel by je měl měnit jen pokud program nefunguje správně!

### Parametry pro nastavení cest k adresářům
- **RAW_IMAGES_PATH**:  Cesta k adresáři s fotografiemi ještěrek.
- **IMAGES_PATH**:  Cesta k adresáři s fotografiemi, která jsou v náhodném pořadí číselně označena a uložena ve formátu png.
- **YOLO_LABELED_PATH**:  Cesta k adresáři s fotografiemi, které vizualizují výsledky detektoru YOLOv4.
- **BODIES_PATH**:  Cesta k adresáři s výřezy ještěrek.
### Parametry pro nastavení cest k datovým souborům
- **PICKLE_PATH**: Název a cesta souboru pickle, ve kterém jsou uloženy informace o zpracovávaných fotografiích.
- **CSV_PATH**:  Název a cesta souboru csv, ve kterém jsou vizualizována data ze souboru pickle.
### Parametry pro nastavení cest ke konfiguračním souborům
- **LABELS_FILE**: Cesta k souboru **obj.names**.
- **CONFIG_FILE**: Cesta k souboru **yolov4-custom.cfg**.
### Parametry pro nastavení cest k modelům
- **WEIGHTS_FILE**: Název a cesta modelu YOLOv4.
- **FIRST_MODEL_PATH**: Název a cesta U-Net modelu 1.
- **SECOND_MODEL_PATH**: Název a cesta U-Net modelu 1.
- **THIRD_MODEL_PATH**: Název a cesta U-Net modelu 1.
### Parametry pro evaluaci modelů
- **CONFIDENCE_THRESHOLD**: Parametr t_bb metody non-maxima suppression.
- **ENLARGE_BODY**: Poměr velikosti vyřezávaného obrazu a velikosti detekovaného břicha.
- **CUT_SIZE**: Rozměry MxN vstupního výřezu do sítě U-Net, kde M = N.
- **LOCAL_FILTER**: Rozměry čtvercového maxfiltru.
- **MINIMUM_DISTANCE**: Minimální přípustná vzdálenost mezi detekovanými body množiny Gamma.
- **TM**: Práh Centroid counterpointu t_m.
- **MAXIMUM_SCALES**: Horní mez počtu šupin ještěrky, pro který program vyhodnotí detekci jako validní.
- **MINIMUM_SCALES**: Spodní mez počtu šupin ještěrky, pro který program vyhodnotí detekci jako validní.

## Popis funkce jednotlivých skriptů
Seznam potřebných knihoven pro spouštění skriptů je v souboru **requirements.txt**.

Do adresáře **dataset_raw** je před spouštěním skriptů potřeba umístit fotografie ventrálních stran ještěrek obecných ve formátech png, jpg, psd nebo CR2. Datová sada těchto fotografií není součástí této ani jiné přílohy.

Spuštění skriptu **prepare.py** transformuje sadu fotografií ve složce **dataset_raw** na stejné fotografie, které jsou v náhodném pořadí číselně označené a uložené ve formátu png do adresáře zadaného parametrem **IMAGES_PATH**. Zároveň vytvoří soubor pickle pro uložení informací pro další skripty.

### Trénování modelu YOLOv4
Výstupy skriptu **prepare.py** je třeba oštítkovat například pomocí programu LabelImg 1.8 pro Windows. Štítkují se třídy head (hlava), tail (ocas) a body (břicho). Fotografie včetně txt štítků a souboru **classes.txt**, který má na každém řádku název jedné štítkované třídy, se přidají do archivu **obj.zip** a vloží do adresáře **google_colab/yolov4/**.

Adresář **google_colab/yolov4/** je před trénováním potřeba nakopírovat do kmenové složky Google Disku. Do adresáře **yolov4/training/** se po ukončení trénování vygeneruje model ve formátu weight. V souborech **obj.data** a **obj.names** jsou nastaveny názvy tříd a polohy ostatních souborů. Skript **process.py** náhodně rozdělí trénovací sadu na trénovací podmnožinu a validační podmnožinu. V konfiguračním souboru **yolov4-custom.cfg** je nastavena architektura sítě a parametry pro trénování. Trénování modelu se spustí příkazy v Jupyter Notebooku **google_colab/yolov4/yolov4_jupyter_train.ipynb**. Jupyter Notebook je nutno spouštět v prostředí služby Google Colaboratory.

Vytrénovaný model se z Google Disku přemístí do adresáře zadaného parametrem **WEIGHTS_FILE** a spuštěním skriptu **cut_body.py** se do adresáře zadaného parametrem **BODIES_PATH** vytvoří výřezy vstupních fotografií ještěrek.

### Trénování modelů U-Net
Ve výřezech vstupních fotografií ještěrek je potřeba oštítkovat šupiny. K tomu slouží skript **labeling.py**. Po spuštění se načtou vstupní výřezy, mezi kterými se přepíná klávesami Q (zpět) a E (další). Levým kliknutím myši se štítkují šupiny, pravým kliknutím se maže poslední zadaná pozice a kolečkem se upravuje poloměr gradientu podle velikosti šupiny. Data se do souboru pickle ukládají průběžně automaticky.

Soubor pickle se umístí do adresáře **google_colab/unet/**. Do složek **google_colab/unet/test_data_images/** a **google_colab/unet/train_data_images/** se umístí výřezy rozdělené na trénovací a testovací sady. Adresář **google_colab/unet/** se umístí do kmenové složky na Google Disku. Trénování modelu se spustí příkazy v Jupyter Notebooku **google_colab/yolov4/unet_jupyter_train.ipynb**. Jupyter Notebook je nutno spouštět v prostředí služby Google Colaboratory. Pro fungování dalších částí je potřeba natrénovat alespoň 3 modely (viz diplomovou práci). Natrénované modely se uloží do adresáře **unet/models** na Google Disku.

### Evaluace modelů
Natrénované modely YOLOv4 a U-Net se umístí do adresáře **models/** a spustí se skript **thesis_evaluation.py**. Tím se do souboru pickle zapíší detekované počty šupin vedle skutečných (štítkovaných). Přehledný csv soubor pak z pickle dat vygeneruje skript **pickle2csv.py**.
