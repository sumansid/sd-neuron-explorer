"""Model loaders with one hook surface: SDX (SDXL-Turbo UNet) and FLX (FLUX.1 schnell DiT)."""


def load_model(name="sdxl", prewarm=()):
    if name == "flux":
        from sd_explorer.flux import FLX
        return FLX(prewarm=prewarm)
    if name == "flux3":
        from sd_explorer.flux3 import FL3
        return FL3(prewarm=prewarm)
    from sd_explorer.model import SDX
    return SDX()
