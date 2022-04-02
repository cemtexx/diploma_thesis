# **ScalesCounter** - cloudová verze
Tento program byl vytvořen v rámci diplomové práce na Ústavu automatizace a informatiky Fakulty strojního inženýrství VUT v Brně. Zároveň slouží vědeckým účelům pro Akademii věd České republiky.

Autor: **Štěpán Maršala**

## Úvod
Program ScalesCounter počítá sekundární šupiny Ještěrek obecných na jejich ventrálních stranách (břichách) a výsledky zapisuje do textového souboru. Program má dvě verze. Verzi pro lokální spuštění na počítači s operačním systém Windows a cloudovou verzi, která se spouští přes službu Google Colab propojitelnou s Google Diskem.

## Návod pro nastavení a spuštění programu
### Na vašem Google Disku
1.   Do hlavní složky Google Disku nakopírujte složku **ScalesCounter/**.
2.   Do adresáře **ScalesCounter/models/** vložte tří natrénované modely sítě U-Net a jeden natrénovaný model sítě YOLOv4. Sada skriptů určená k trénování těchto modelů je v jiné příloze. Modely nejsou součástí této ani jiné přílohy.
3.   Do adresáře **ScalesCounter/images_input/** vložte fotografie ještěrek ve formátech jpg, psd, png nebo CR2. Sada fotografií není součástí této ani jiné přílohy.

### Na Google Colaboratory
1.   Jděte na stránky https://colab.research.google.com/ a přihlaste se pod stejným Google účtem jako na Google Disk v předchozích instrukcích.
2.   Nakopírujte sešit **ScalesCounter/ScalesCounter.ipynb** do výchozího adresáře nebo kteréhokoliv jiného, v případě problémů s importováním souboru .ipynb vytvořte prázdný sešit a do něj nakopírujte jednotlivá okna souboru **ScalesCounter.ipynb**.
3.   Sešit otevřete.
4.   V záložce **Runtime** zvolte **Change runtime type** a vyberte buď **GPU** nebo **None** podle toho, zda je kapacita na akcelerátoru volná (Google Colab upozorní, pokud nebude).
5.   Postupně spouštějte jednotlivá okna sešitu.

### V Jupyter Notebooku
1.   První okno nainstaluje dodatečné knihovny pro Python nutné pro běh programu.
2.   Druhým oknem se připojíte ke Google Disku, je potřeba zadat ověřovací kód, na který budete přesměrováni.
3.   Třetím oknem nastavíte parametry. Výchozí hodnoty jsou však ozkoušené a uživatel by je měl měnit jen pokud program nefunguje správně!

Vysvětlení parametrů:
- **IMAGE_OUTPUT**:  Pokud je nastavena hodnota True, program bude do kmenové složky s programem vracet vizualizaci výstupů v podobě výřezů ještěrek s vyznačenými šupinami. Hodnota False tuto funkci vypíná.
- **FINISH_UP**: False pro novou úlohu nebo název předchozí úlohy (job name) v 'uvozovkách' pro její dokončení.
- **INPUT_FOLDER**: Cesta k adresáři se vstupy.  
- **OUTPUT_FOLDER**: Cesta k adresáři s vizualizovanými výstupy a textovým souborem. 
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

4.   Čtvrté okno spustí program.
   - Vyhodnocení jedné fotografie může trvat i několik vteřin.
   - Program se přeruší stisknutím symbolu pro přerušení v levém horním rohu okna.
   - Všechny výsledky jsou průběžně zaznamenávány a uživatel je může sledovat v průběhu běhu.
   - Pokud dojde z jakéhokoliv důvodu k přerušení běhu, parametr **FINISH_UP** změňte na název jobu (časovou značku, název složek a txt výstupu) podle instrukcí výše a program spusťte znova v záložce **Runtime** kliknutím na **Restart and run all**.

### Výsledky
1.   Výsledky najdete ve složce vašeho Google Disku **ScalesCounter/image_output**, která byla vytvořena ve složce s programem, jako obrázky a txt soubor.
