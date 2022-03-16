Ez a program az adásvételi szerződések beolvasásáért felelős alkalmazás szerver komponense.

A futtatáshoz az alábbi környezet igényeltetik:
	- Microsoft SQL Server Database Engine (VS tartalmazza)
	- Python 3.8
	- transformers Python csomag (pip install transformers)


A futtatáshoz az AdasVetelServer.sln fájlt kell megnyitni
Microsoft Visual Studióval és futtatni annak segítségével.


A futtatás előtt javasolt az alábbi configurációkat elvégezni:

- az AdasVetelServer\config\dbServerPath.conf fájl tartalmazza az adatbázis szerver nevét.
Ez alapértelmezetten (localdb)\MSSQLLocalDB. Ezt a kipróbálás alatt érdemes így hagyni és
az MSSQL szervert ezen a néven létrehozni.

- AdasVetelServer\config\pythonPath.loc tartalmazza a telepített Python.exe futtatható állomány teljes
abszolút elérési útvonalát fájlnévvel együtt. Ezt a NER modell használatához át kell írni a gépen található
Python.exe elérési útvonalára.

A program minden fordításkor új, üres adatbázist hoz létre, ezért első fordításkor a szerveren regisztrálni kell
egy új felhasználót, amivel a kliens oldalon be lehet majd lépni.



