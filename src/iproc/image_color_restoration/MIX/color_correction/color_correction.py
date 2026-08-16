from iproc.image_color_restoration.MIX.color_correction.adaptive_stretch_lab import LAB_Stretching

class ColorCorrection():
    @classmethod
    def color_correct(cls, image):
        return LAB_Stretching(image)


