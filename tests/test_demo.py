"""Demo video for the defence (session 05): the frame with the incident state, risk and the Telegram card."""

import numpy as np

from sentinel.demo import PANEL_W, compose


def test_the_demo_frame_has_the_video_a_state_banner_and_a_card_panel():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = compose(
        frame, state="тривога", risk=0.82, decided_by="A", t=13.0, card="🔴 Ймовірна бійка — Хол\n15:08:09"
    )
    assert out.shape == (480, round(320 * 480 / 240) + PANEL_W, 3)
    banner = out[:40, : out.shape[1] - PANEL_W]
    assert banner[..., 2].mean() > banner[..., 1].mean()  # an alarm banner is red (BGR)
    panel = out[:, -PANEL_W:]
    assert panel.std() > 5  # the card is drawn


def test_a_calm_frame_has_a_green_banner_and_an_empty_panel():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = compose(frame, state="норма", risk=0.1, decided_by="A", t=1.0, card=None)
    banner = out[:40, : out.shape[1] - PANEL_W]
    assert banner[..., 1].mean() > banner[..., 2].mean()
