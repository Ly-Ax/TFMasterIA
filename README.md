<img src="images/viu_cabecera.webp" width="200px">

# Máster en Inteligencia Artificial

## TFM: Comparación de Algoritmos de DL frente a Algoritmos de ML Clásicos 
## Caso: Aprobación de Préstamos de la U.S. Small Business Administration (SBA)

[**Alex Castro Gumiel**](https://www.linkedin.com/in/alex-castro-gumiel/)

> **Conjunto de Datos**

https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied

**Contexto:**

El conjunto de datos es de la Administración de Pequeñas Empresas de EE.UU. (SBA). La SBA de EE.UU. se fundó en 1953 con el principio de promover y ayudar a las pequeñas empresas en el mercado crediticio de EE.UU. Las pequeñas empresas han sido una fuente principal de creación de empleo en los Estados Unidos; por lo tanto, fomentar la formación y el crecimiento de pequeñas empresas tiene beneficios sociales al crear oportunidades laborales y reducir el desempleo.

Ha habido muchas historias de éxito de empresas emergentes que recibieron garantías de préstamos de la SBA, como FedEx y Apple Computer. Sin embargo, también ha habido historias de pequeñas empresas y/o nuevas empresas que han incumplido con sus préstamos garantizados por la SBA.

**Diccionario de Datos:**

Contiene 899164 instancias y 27 variables.

|Número|Variable|Descripción|
|------|--------|-----------|
|1|LoanNr_ChkDgt|Identificador - Clave principal|
|2|Name|Nombre del prestatario|
|3|City|Ciudad del prestataria|
|4|State|Estado del prestatario|
|5|Zip|Código postal del prestatario|
|6|Bank|Nombre del banco|
|7|BankState|Estado del banco|
|8|NAICS|Código del sistema de clasificación de la industria de América del Norte|
|9|ApprovalDate|Fecha de emisión del compromiso de la SBA|
|10|ApprovalFY|Año fiscal del compromiso|
|11|Term|Plazo del préstamo en meses|
|12|NoEmp|Número de empleados de la empresa|
|13|NewExist|1 = Negocio existente, 2 = Nuevo negocio|
|14|CreateJob|Número de trabajos creados|
|15|RetainedJob|Número de trabajos retenidos|
|16|FranchiseCode|Código de franquicia, (00000 o 00001) = Sin franquicia|
|17|UrbanRural|1 = Urbano, 2 = rural, 0 = indefinido|
|18|RevLineCr|Línea de crédito renovable: Y = Sí, N = No|
|19|LowDoc|Programa de préstamos: Y = Sí, N = No|
|20|ChgOffDate|La fecha en que un préstamo se declara en mora|
|21|DisbursementDate|Fecha de desembolso|
|22|DisbursementGross|Monto Bruto desembolsado|
|23|BalanceGross|Saldo Bruto pendiente|
|24|MIS_Status|Estado del préstamo cancelado = CHGOFF, Pagado en su totalidad = PIF|
|25|ChgOffPrinGr|Importe cancelado|
|26|GrAppv|Importe bruto del préstamo aprobado por el banco|
|27|SBA_Appv|Monto garantizado del préstamo aprobado por la SBA|

Descripción de los dos primeros dígitos del NAICS:

|Sector|Descripción|
|------|-----------|
|11|Agricultura, silvicultura, pesca y caza|
|21|Minería, explotación de canteras y extracción de petróleo y gas|
|22|Utilidades|
|23|Construcción|
|31-33|Manufactura|
|42|Comercio al por mayor|
|44-45|Comercio minorista|
|48-49|Transporte y almacenamiento|
|51|Información|
|52|Finanzas y seguros|
|53|Bienes inmuebles y alquiler y arrendamiento|
|54|Servicios profesionales, científicos y técnicos|
|55|Gestión de sociedades y empresas|
|56|Servicios administrativos y de apoyo y gestión de residuos y remediación|
|61|Servicios educativos|
|62|Asistencia sanitaria y asistencia social|
|71|Artes, entretenimiento y recreación|
|72|Servicios de alojamiento y alimentación|
|81|Otros servicios (excepto administración pública)|
|92|Administración pública|

**Agradecimientos:**

Conjunto de datos original: "Should This Loan be Approved or Denied?”: A Large Dataset with Class Assignment Guidelines", por: Min Li, Amy Mickel & Stanley Taylor.

Enlace al artículo: https://doi.org/10.1080/10691898.2018.1434342

> **Estructura del Proyecto**

    ├── data
        ├── clean
        ├── raw
            ├── sba_national.csv -> Dataset original [DVC - GCP]
    ├── docs
        ├── sba_guidelines_en.pdf -> Articulo original (Inglés)
        ├── sba_guidelines_es.pdf -> Articulo traducido (Español)
    ├── images
        ├── viu_cabecera.webp -> Logotipo de la VIU para cabeceras
    ├── models
    ├── notebooks
        ├── data_exploration.ipynb -> Análisis de Datos Exploratorio
    ├── src
        ├── classifier
        ├── load
        ├── train
        ├── transform

    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── requirements.txt

<!-- ```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
``` -->
