# Comprensión de Datos

Extraido del artículo: https://doi.org/10.1080/10691898.2018.1434342

## Antecedentes

La Administración de Pequeñas Empresas (SBA) de EE.UU. se fundó en 1953 con el principio de promover y ayudar a las pequeñas empresas en el mercado  crediticio de Estados  Unidos. Las pequeñas empresas han sido una fuente principal de creación de empleo en los Estados Unidos; por lo tanto, fomentar la formación y el crecimiento de pequeñas empresas tiene beneficios sociales al crear oportunidades de trabajo y reducir el desempleo. Una forma en que la SBA ayuda a estas  pequeñas empresas es a través de un programa de garantía de préstamo que está diseñado para alentar a los bancos a otorgar préstamos a las pequeñas empresas. La SBA actúa como un proveedor de seguros para reducir el riesgo de un banco al asumir parte del riesgo garantizando una parte del préstamo. En el caso de que un préstamo entre en incumplimiento, la SBA cubre el monto garantizado.

Ha habido muchas historias de éxito de empresas emergentes que recibieron garantías de préstamos de la SBA, como FedEx y Apple Computer. Sin embargo, también ha habido historias de pequeñas empresas y/o nuevas empresas que han incumplido con sus préstamos garantizados por la SBA. La tasa de morosidad de estos préstamos ha sido motivo de controversia durante décadas. Los economistas conservadores creen que los mercados de crédito funcionan de manera eficiente sin la participación del gobierno. Los partidarios de los préstamos garantizados por la SBA argumentan que los beneficios sociales de la creación de empleo por parte de las pequeñas empresas que reciben préstamos garantizados por el gobierno superan con creces los costos incurridos por los préstamos en mora.

## Descripción del Dataset

|Variable|Data type|Descripción|
|--------|---------|-----------|
|LoanNr_ChkDgt|Text|Identificador - Clave principal|
|Name|Text|Nombre del prestatario|
|City|Text|Ciudad del prestataria|
|State|Text|Estado del prestatario|
|Zip|Text|Código postal del prestatario|
|Bank|Text|Nombre del banco|
|BankState|Text|Estado del banco|
|NAICS|Text|Código del sistema de clasificación de la industria de América del Norte|
|ApprovalDate|Date/Time|Fecha de emisión del compromiso de la SBA|
|ApprovalFY|Text|Año fiscal del compromiso|
|Term|Number|Plazo del préstamo en meses|
|NoEmp|Number|Número de empleados de la empresa|
|NewExist|Text|1 = Negocio existente, 2 = Nuevo negocio|
|CreateJob|Number|Número de trabajos creados|
|RetainedJob|Number|Número de trabajos retenidos|
|FranchiseCode|Text|Código de franquicia: (00000 o 00001) = Sin franquicia|
|UrbanRural|Text|1 = Urbano, 2 = Rural, 0 = Indefinido|
|RevLineCr|Text|Línea de crédito renovable: Y = Si, N = No|
|LowDoc|Text|Programa de préstamos: Y = Si, N = No|
|ChgOffDate|Date/Time|La fecha en que un préstamo se declara en mora|
|DisbursementDate|Date/Time|Fecha de desembolso|
|DisbursementGross|Currency|Monto Bruto desembolsado|
|BalanceGross|Currency|Saldo Bruto pendiente|
|MIS_Status|Text|Estado del préstamo cancelado = CHGOFF, Pagado en su totalidad = PIF|
|ChgOffPrinGr|Currency|Importe cancelado|
|GrAppv|Currency|Importe bruto del préstamo aprobado por el banco|
|SBA_Appv|Currency|Monto garantizado del préstamo aprobado por la SBA|

**NAICS** (Sistema de Clasificación de la Industria de América del Norte): 

Este es un sistema de clasificación jerárquico de 2 a 6 dígitos utilizado por las agencias estadísticas federales para clasificar los establecimientos comerciales para la recopilación, el análisis y la presentación de información. Los dos primeros dígitos de la clasificación NAICS representan el sector económico.

|Sector|Descripción|
|------|-----------|
|11|Agricultura, silvicultura, pesca y caza|
|21|Minería, explotación de canteras, y extracción de petróleo y gas|
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

**NewExist** (1 = Negocio Existente, 2 = Nuevo Negocio): 

Esto representa si el negocio es un negocio existente (en existencia por más de 2 años) o un nuevo negocio (en existencia por menos de o igual a 2 años).

**LowDoc** (Y = Si, N = No):  

Para poder procesar préstamos de manera más eficiente, se implementó un programa de “Préstamo LowDoc” en el que se pueden procesar préstamos de menos de $150,000 mediante una solicitud de una página. “Si” indica préstamos con una solicitud de una página y “No” indica préstamos con más información adjunta a la solicitud. 

En este conjunto de datos, el 87.31% está codificado como N (No) y el 12.31% como Y (Si), para un total de 99.62%. Cabe resaltar que el 0.38% tiene otros valores (0, 1, A, C, R, S); estos son errores de entrada de datos. También hay 2582 valores perdidos para esta variable, excluidos al calcular estas proporciones.

**MIS_Status**:

Esta variable indica el estado del préstamo: en mora/caído (CHGOFF) o bien pagado en su totalidad (PIF).

## Periodo de Tiempo

Se considera que la inclusión de préstamos con fechas de desembolso posteriores a 2010 daría mayor peso a aquellos préstamos que se cancelan frente a los que se pagan en su totalidad. Más específicamente, los préstamos cancelados lo harán antes de la fecha de vencimiento del préstamo, mientras que los préstamos que probablemente se pagarán en su totalidad lo harán en la fecha de vencimiento del préstamo (que se extendería más allá del conjunto de datos que finaliza en 2014). Dado que este conjunto de datos se ha restringido a los préstamos para los que se conoce el resultado, existe una mayor probabilidad de que los préstamos cancelados antes de la fecha de vencimiento se incluyan en el conjunto de datos, mientras que los que podrían pagarse en su totalidad han sido excluidos.
