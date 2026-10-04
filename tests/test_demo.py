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


def test_hidden_boxes_are_blurred_in_the_output_and_the_rest_is_untouched():
    import numpy as np

    from sentinel.demo import hide

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    out = hide(image, [[0.0, 0.0, 0.25, 0.1]])
    assert (out[10:] == image[10:]).all() and (out[:10, 50:] == image[:10, 50:]).all()
    assert out[:10, :50].std() < image[:10, :50].std()  # blurred
    assert (hide(image, []) == image).all()


def test_chained_sources_follow_each_other_in_time(tmp_path):
    from conftest import make_video

    from sentinel.demo import chained

    a = make_video(tmp_path / "a.avi", n_frames=30, fps=30)  # 1 s
    b = make_video(tmp_path / "b.avi", n_frames=30, fps=30)
    frames = list(chained([str(a), str(b)], fps=10))
    times = [f.t for f in frames]
    assert len(frames) == 20 and [f.index for f in frames] == list(range(20))
    assert all(t2 > t1 for t1, t2 in zip(times, times[1:], strict=False))  # never backwards across the join
    assert abs(times[10] - (times[9] + 0.1)) < 1e-6
