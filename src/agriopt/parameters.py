from __future__ import annotations
import numpy as np

CROP_ORDER=[
'Ahuyama','Arveja verde','Cebolla cabezona','Cebolla larga','Cilantro','Cebada','Maiz Blanco','Frijol','Habichuela','Lechuga','Perejil','Pimenton','Tomate','Banano','Lulo','Papaya','Pina','Arracacha','Papas','Platano','Yuca']
MATURITY_WEEKS={
'Ahuyama':15,'Arveja verde':13,'Cebolla cabezona':20,'Cebolla larga':11,'Cilantro':9,'Cebada':10,'Maiz Blanco':11,'Frijol':15,'Habichuela':14,'Lechuga':9,'Perejil':13,'Pimenton':5,'Tomate':18,'Banano':56,'Lulo':55,'Papaya':44,'Pina':78,'Arracacha':52,'Papas':20,'Platano':56,'Yuca':64}
BOTANICAL={
'Ahuyama':'H','Arveja verde':'H','Cebolla cabezona':'H','Cebolla larga':'H','Cilantro':'H','Cebada':'G','Maiz Blanco':'C','Frijol':'H','Habichuela':'H','Lechuga':'H','Perejil':'H','Pimenton':'H','Tomate':'H','Banano':'F','Lulo':'F','Papaya':'F','Pina':'F','Arracacha':'T','Papas':'T','Platano':'F','Yuca':'T'}
SALES_GROUP_NAME={
'Ahuyama':'VH','Arveja verde':'VH','Cebolla cabezona':'VH','Cebolla larga':'VH','Cilantro':'VH','Cebada':'OG','Maiz Blanco':'OG','Frijol':'VH','Habichuela':'VH','Lechuga':'VH','Perejil':'VH','Pimenton':'VH','Tomate':'VH','Banano':'F','Lulo':'F','Papaya':'F','Pina':'F','Arracacha':'TRP','Papas':'TRP','Platano':'F','Yuca':'TRP'}
SALES_GROUP_ID={'VH':0,'F':1,'TRP':2,'OG':3}
PRODUCT_SETS={
5:['Yuca','Maiz Blanco','Platano','Tomate','Pina'],
8:['Yuca','Maiz Blanco','Platano','Tomate','Pina','Papaya','Papas','Arveja verde'],
12:['Yuca','Maiz Blanco','Platano','Tomate','Pina','Papaya','Papas','Arveja verde','Perejil','Arracacha','Lulo','Habichuela'],
21:CROP_ORDER.copy(),
}
ALIASES={'Cebolla Junca':'Cebolla larga','Maiz':'Maiz Blanco','Papa':'Papas','Piña':'Pina','Pinna':'Pina'}
