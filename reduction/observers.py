from astropy.coordinates import EarthLocation


bernd = EarthLocation.from_geodetic(13.708425, 51.003557, 270)
bernd.info.name = "Dresden, Germany"

christian = EarthLocation.from_geodetic(13.7, 51, 300)
christian.info.name = r"Gönnsdorf"

bernd = EarthLocation.from_geodetic(13.708425, 51.003557, 270)
bernd.info.name = "Dresden, Germany"

ulrich = EarthLocation.from_geodetic(13.5, 52.5, 100)
ulrich.info.name = "Berlin Marzahn, Germany"

filipe = EarthLocation.from_geodetic(-8.365, +37.132)
filipe.info.name = r"Armação de Pêra, Portugal"
