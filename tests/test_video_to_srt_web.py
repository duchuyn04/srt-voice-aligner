import tempfile
import unittest
from pathlib import Path

from video_to_srt_web import Segment, merge_segments, ms_to_srt_timestamp, write_srt


class VideoToSrtUtilityTests(unittest.TestCase):
    def test_ms_to_srt_timestamp(self) -> None:
        self.assertEqual(ms_to_srt_timestamp(0), "00:00:00,000")
        self.assertEqual(ms_to_srt_timestamp(3_661_234), "01:01:01,234")

    def test_merge_segments_offsets_and_drops_overlap_duplicate(self) -> None:
        chunks = [
            [Segment(0, 1200, "hello"), Segment(1200, 2500, "world")],
            [Segment(2300, 2600, "world"), Segment(2600, 4000, "again")],
        ]
        merged = merge_segments(chunks, overlap_tolerance_ms=250)
        self.assertEqual([item.text for item in merged], ["hello", "world", "again"])
        self.assertEqual(merged[2].start_ms, 2600)

    def test_write_srt_uses_contiguous_indexes_and_timelines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.srt"
            write_srt(
                output,
                [
                    Segment(0, 1000, "First"),
                    Segment(1250, 2500, "Second"),
                ],
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n"
                "2\n00:00:01,250 --> 00:00:02,500\nSecond\n",
            )


if __name__ == "__main__":
    unittest.main()
