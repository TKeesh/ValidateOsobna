# ValidateOsobna

## Requirements

Python

OS - uz python

DateTime - uz python

Numpy
```Shell
pip install numpy
```

OpenCV 3
```Shell
pip install opencv-python
```

## Instalacija

```Shell
git clone https://github.com/TKeesh/ValidateOsobna.git
```

## Priprema za test

U direktorij `./test/` staviti slike (.jpg) za testiranje

## Pokretanje

```Shell
./python validate.py
```

`anyKey` - sljedeća slika

`q` - quit

## Opis

Skripta ide po svakoj .jpg slici u test direktoriju i provjerava da li je na njoj prednja strana osobne.
Pretpostavka je da će slike biti otprilike rezolucije 385x240. Manje odstupanje ne bi trebalo predstavljati problem (+- 25%), za nešto značajnije problem su hardkoridani thresholdi, ali algoritam radi. Za takav slučaj treba podesiti vrijednosti u validate_front() funkciji. 

## Note: 
**Dobio sam sljedeći ispis za endline convert prilikom pushanja na git, nadam se da će raditi:**\
warning: LF will be replaced by CRLF in haarcascade_frontalface.xml.\
The file will have its original line endings in your working directory.