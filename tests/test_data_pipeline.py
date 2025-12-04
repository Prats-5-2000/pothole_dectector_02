#!/usr/bin/env python3
"""
Unit Tests for Data Pipeline

Tests frame extraction, mask conversion, and YOLO label generation.

Author: Pothole Detection Team
Date: 2025-12-04
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline import (
    mask_to_yolo_boxes,
    voc_xml_to_yolo,
    extract_frames_from_video,
    save_yolo_labels
)


class TestMaskToYolo(unittest.TestCase):
    """Test mask to YOLO conversion."""
    
    def test_single_region(self):
        """Test conversion with single detection region."""
        # Create synthetic mask with one white region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 255  # 20x30 box
        
        boxes = mask_to_yolo_boxes(mask, min_area=100, class_id=0)
        
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][0], 0)  # Class ID
        
        # Check normalized coordinates
        x_center = boxes[0][1]
        y_center = boxes[0][2]
        width = boxes[0][3]
        height = boxes[0][4]
        
        self.assertGreater(x_center, 0)
        self.assertLess(x_center, 1)
        self.assertGreater(y_center, 0)
        self.assertLess(y_center, 1)
        self.assertGreater(width, 0)
        self.assertLess(width, 1)
        self.assertGreater(height, 0)
        self.assertLess(height, 1)
    
    def test_multiple_regions(self):
        """Test conversion with multiple detection regions."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # Region 1
        mask[60:80, 60:80] = 255  # Region 2
        
        boxes = mask_to_yolo_boxes(mask, min_area=100)
        
        self.assertEqual(len(boxes), 2)
    
    def test_min_area_filter(self):
        """Test that small regions are filtered out."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:15] = 255  # Small region (25 pixels)
        mask[50:80, 50:80] = 255  # Large region (900 pixels)
        
        boxes = mask_to_yolo_boxes(mask, min_area=100)
        
        # Only large region should be detected
        self.assertEqual(len(boxes), 1)
    
    def test_empty_mask(self):
        """Test with empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        boxes = mask_to_yolo_boxes(mask, min_area=100)
        
        self.assertEqual(len(boxes), 0)


class TestFrameExtraction(unittest.TestCase):
    """Test frame extraction from videos."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_frame_naming(self):
        """Test that frames are named correctly."""
        # Create a small test video
        video_path = self.temp_path / "test_video.mp4"
        output_dir = self.temp_path / "frames"
        
        # Create test video (10 frames, 10 FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 10, (320, 240))
        
        for i in range(10):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Extract frames
        extracted = extract_frames_from_video(video_path, output_dir, fps=1)
        
        # Check naming pattern
        self.assertGreater(len(extracted), 0)
        
        for frame_path in extracted:
            # Should match pattern: test_video_frame_000000.jpg
            self.assertTrue(frame_path.name.startswith("test_video_frame_"))
            self.assertTrue(frame_path.suffix == ".jpg")
            
            # Check frame exists
            self.assertTrue(frame_path.exists())


class TestYoloLabels(unittest.TestCase):
    """Test YOLO label saving and format."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_labels(self):
        """Test saving YOLO format labels."""
        boxes = [
            [0, 0.5, 0.5, 0.3, 0.2],
            [0, 0.7, 0.3, 0.15, 0.25]
        ]
        
        label_path = self.temp_path / "test_label.txt"
        save_yolo_labels(boxes, label_path)
        
        self.assertTrue(label_path.exists())
        
        # Read and verify format
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        # Check first line format
        parts = lines[0].strip().split()
        self.assertEqual(len(parts), 5)  # class, x, y, w, h
        
        # Check values are floats
        for part in parts:
            float(part)  # Should not raise exception


class TestVocXmlConversion(unittest.TestCase):
    """Test Pascal VOC XML to YOLO conversion."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_voc_conversion(self):
        """Test VOC XML to YOLO conversion."""
        # Create sample XML
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotation>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>pothole</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>200</ymax>
        </bndbox>
    </object>
</annotation>
"""
        
        xml_path = self.temp_path / "test.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        
        boxes = voc_xml_to_yolo(xml_path, class_id=0)
        
        self.assertEqual(len(boxes), 1)
        
        # Check format
        self.assertEqual(boxes[0][0], 0)  # Class ID
        
        # Check normalized coordinates
        x_center = boxes[0][1]
        y_center = boxes[0][2]
        width = boxes[0][3]
        height = boxes[0][4]
        
        self.assertGreater(x_center, 0)
        self.assertLess(x_center, 1)
        self.assertGreater(y_center, 0)
        self.assertLess(y_center, 1)
        
        # Expected values for 100,100,200,200 box in 640x480 image
        # x_center = 150/640 = 0.234375
        # y_center = 150/480 = 0.3125
        # width = 100/640 = 0.15625
        # height = 100/480 = 0.208333
        
        self.assertAlmostEqual(x_center, 0.234375, places=4)
        self.assertAlmostEqual(y_center, 0.3125, places=4)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMaskToYolo))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestYoloLabels))
    suite.addTests(loader.loadTestsFromTestCase(TestVocXmlConversion))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
