# **ScalesCounter** - lokální verze pro Windows
Tento program byl vytvořen v rámci diplomové práce na Ústavu automatizace a informatiky Fakulty strojního inženýrství VUT v Brně. Zároveň slouží vědeckým účelům pro Akademii věd České republiky.

Autor: **Štěpán Maršala**

## Úvod
Program ScalesCounter počítá sekundární šupiny Ještěrek obecných na jejich ventrálních stranách (břichách) a výsledky zapisuje do textového souboru. Program má dvě verze. Verzi pro lokální spuštění na počítači s operačním systém Windows a cloudovou verzi, která se spouští přes službu Google Colab propojitelnou s Google Diskem.

## Návod pro spuštění programu
Program je nejprve potřeba sestavit (nainstalovat), nastavit a až pak se může spustit.

### Sestavení programu
1.   Do adresáře **models/** vložte tří natrénované modely sítě U-Net a jeden natrénovaný model sítě YOLOv4. Jejich názvy musí odpovídat parametrům **FIRST_MODEL_PATH**, **SECOND_MODEL_PATH**, **THIRD_MODEL_PATH** a **YOLO_WEIGHTS_FILE** v souborech **config.yaml** a **setup.py**. Sada skriptů určená k trénování těchto modelů je v jiné příloze. Modely nejsou součástí této ani jiné přílohy.
2. Nainstalujte potřebné Python knihovny, které jsou vypsané v adresáři **requirements.txt**.
3. Sestavte program zadáním následujícího příkazu do příkazového řádku:
```console
python setup.py build
```
Ve výchozím adresáři se objeví složka se sestaveným programem. Ve složce s programem uživatel ovládá pouze soubory **config.yaml** a **main.exe**.

### Nastavení parametrů
V nové složce s programem najdete soubor **config.yaml**, který je editovatelný v běžných textových editorech jako je například poznámkový blok ve Windows. Slouží k nastavování parametrů programu. Jeho výchozí hodnoty jsou však vyzkoušené a uživatel by je měl měnit jen pokud program nefunguje správně!

Vysvětlení parametrů
- **IMAGE_OUTPUT**:  Pokud je nastavena hodnota True, program bude do kmenové složky s programem vracet vizualizaci výstupů v podobě výřezů ještěrek s vyznačenými šupinami. Hodnota False tuto funkci vypíná.
- **DEBUG_MODE**: Pokud je nastavena hodnota True, tak bude docházet k výpisu logů knihovny tensorflow.
- **IMAGES_INPUT_FOLDER**: Cesta k adresáři se vstupy.  
- **IMAGES_OUTPUT_FOLDER**: Cesta k adresáři se vstupy s vizualizovanými výstupy. 
- **TXT_OUTPUT_FOLDER**: Cesta k adresáři s textovými výstupy.
- **YOLO_CONFIG_FILE**: Cesta ke konfiguračnímu souboru pro síť YOLOv4.
- **YOLO_WEIGHTS_FILE**: Cesta k modelu sítě YOLO.
- **FIRST_MODEL_PATH**: Cesta k prvnímu modelu sítě U-Net.
- **SECOND_MODEL_PATH**: Cesta ke druhému modelu sítě U-Net.
- **THIRD_MODEL_PATH**: Cesta ke třetímu modelu sítě U-Net.
- **TEMPORARY_IMAGE**: Název dočasného souboru pro ladění.
- **CONFIDENCE_THRESHOLD**: Parametr t_bb metody non-maxima suppression.
- **ENLARGE_BODY**: Poměr velikosti vyřezávaného obrazu a velikosti detekovaného břicha.
- **CUT_SIZE**: Rozměry MxN vstupního výřezu do sítě U-Net, kde M = N.
- **LOCAL_FILTER**: Rozměry čtvercového maxfiltru.
- **MINIMUM_DISTANCE**: Minimální přípustná vzdálenost mezi detekovanými body množiny Gamma.
- **TM**: Práh centroid counterpointu t_m.
- **MAXIMUM_SCALES**: Horní mez počtu šupin ještěrky, pro který program vyhodnotí detekci jako validní.
- **MINIMUM_SCALES**: Spodní mez počtu šupin ještěrky, pro který program vyhodnotí detekci jako validní.

### Spuštění
1.   Do složky **images_input/** v adresáři s programem vložte fotografie ještěrek ve formátech jpg, psd, png nebo CR2. Sada fotografií není součástí této ani jiné přílohy.
2.   Ve složce **models/** v adresáři s programem můžete nahradit modely jinými.
3.   Spuštěním program spuštěním souboru **main.exe**. Vyhodnocení jedné fotografie může trvat i několik vteřin. Program se přeruší stisknutím kláves CTRL+C v okně programu. Všechny výsledky jsou průběžně zaznamenávány a můžete je sledovat v průběhu běhu.

### Výsledky
1.   Výsledky najdete ve složce pojmenované parametrem **IMAGES_OUTPUT_FOLDER** v adresáři se sestaveným programem.
