"""Main CLI entry point for Cat Watcher."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cat Watcher - ML-powered cat behavior detection",
        prog="cat-watcher",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Label command
    label_parser = subparsers.add_parser(
        "label",
        help="Run labeling web UI",
    )
    label_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    label_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to",
    )
    label_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Web UI command (unified interface)
    web_parser = subparsers.add_parser(
        "web",
        help="Run unified web UI (recommended)",
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to",
    )
    web_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML models",
    )
    train_subparsers = train_parser.add_subparsers(dest="train_type", help="Model type")

    # Train behavior model
    train_behavior = train_subparsers.add_parser(
        "behavior",
        help="Train behavior detection model (YOLOv8)",
    )
    train_behavior.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with YOLO-format training data",
    )
    train_behavior.add_argument(
        "--output-dir",
        type=str,
        default="runs/detect",
        help="Output directory for model",
    )
    train_behavior.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_behavior.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    train_behavior.add_argument(
        "--model-size",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    train_behavior.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on (e.g., 'cpu', '0', '0,1')",
    )
    train_behavior.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export after training",
    )

    # Train cat ID model
    train_catid = train_subparsers.add_parser(
        "catid",
        help="Train cat identification model (EfficientNet)",
    )
    train_catid.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with classification training data",
    )
    train_catid.add_argument(
        "--output-dir",
        type=str,
        default="runs/catid",
        help="Output directory for model",
    )
    train_catid.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    train_catid.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    train_catid.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        help="Model architecture (efficientnet_b0 to b7)",
    )
    train_catid.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on",
    )
    train_catid.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX export after training",
    )

    # Prepare data command
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare labeled data for training",
    )
    prepare_subparsers = prepare_parser.add_subparsers(dest="prepare_type", help="Data type")

    # Prepare behavior data
    prepare_behavior = prepare_subparsers.add_parser(
        "behavior",
        help="Prepare data for behavior model",
    )
    prepare_behavior.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to labeling SQLite database",
    )
    prepare_behavior.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for prepared data",
    )
    prepare_behavior.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )

    # Prepare catid data
    prepare_catid = prepare_subparsers.add_parser(
        "catid",
        help="Prepare data for cat ID model",
    )
    prepare_catid.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to labeling SQLite database",
    )
    prepare_catid.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for prepared data",
    )
    prepare_catid.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )

    # Inference command
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference API or single image inference",
    )
    inference_subparsers = inference_parser.add_subparsers(
        dest="inference_type",
        help="Inference mode",
    )

    # Inference API server
    inference_api = inference_subparsers.add_parser(
        "api",
        help="Run inference API server",
    )
    inference_api.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    inference_api.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    inference_api.add_argument(
        "--behavior-model",
        type=str,
        help="Path to behavior detection model",
    )
    inference_api.add_argument(
        "--catid-model",
        type=str,
        help="Path to cat identification model",
    )
    inference_api.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run inference on",
    )
    inference_api.add_argument(
        "--no-onnx",
        action="store_true",
        help="Use PyTorch instead of ONNX",
    )
    inference_api.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # Single image inference
    inference_image = inference_subparsers.add_parser(
        "image",
        help="Run inference on a single image",
    )
    inference_image.add_argument(
        "image_path",
        type=str,
        help="Path to image file",
    )
    inference_image.add_argument(
        "--behavior-model",
        type=str,
        help="Path to behavior detection model",
    )
    inference_image.add_argument(
        "--catid-model",
        type=str,
        help="Path to cat identification model",
    )
    inference_image.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run inference on",
    )
    inference_image.add_argument(
        "--no-onnx",
        action="store_true",
        help="Use PyTorch instead of ONNX",
    )
    inference_image.add_argument(
        "--output",
        type=str,
        help="Save annotated image to path",
    )

    # Home Assistant command
    ha_parser = subparsers.add_parser(
        "homeassistant",
        aliases=["ha"],
        help="Home Assistant integration commands",
    )
    ha_subparsers = ha_parser.add_subparsers(dest="ha_type", help="HA command type")

    # HA discover command
    ha_discover = ha_subparsers.add_parser(
        "discover",
        help="Publish MQTT auto-discovery messages",
    )
    ha_discover.add_argument(
        "--topic-prefix",
        type=str,
        default="cat_watcher",
        help="MQTT topic prefix",
    )
    ha_discover.add_argument(
        "--device-name",
        type=str,
        default="Cat Watcher",
        help="Device name in Home Assistant",
    )
    ha_discover.add_argument(
        "--remove",
        action="store_true",
        help="Remove entities instead of creating",
    )

    # HA status command
    ha_status = ha_subparsers.add_parser(
        "status",
        help="Publish service status",
    )
    ha_status.add_argument(
        "--online",
        action="store_true",
        default=True,
        help="Publish online status (default)",
    )
    ha_status.add_argument(
        "--offline",
        action="store_true",
        help="Publish offline status",
    )

    # HA test command
    ha_test = ha_subparsers.add_parser(
        "test",
        help="Publish a test detection event",
    )
    ha_test.add_argument(
        "--behavior",
        type=str,
        default="cat_eating",
        help="Behavior type to simulate",
    )
    ha_test.add_argument(
        "--cat",
        type=str,
        default="starbuck",
        help="Cat name to simulate",
    )
    ha_test.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Detection confidence",
    )

    # Detect command (stream processing)
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detection pipeline commands (stream processing)",
    )
    detect_subparsers = detect_parser.add_subparsers(
        dest="detect_type", help="Detection command"
    )

    # detect test-stream: Test RTSP stream reading
    detect_test_stream = detect_subparsers.add_parser(
        "test-stream",
        help="Test RTSP stream reading from a camera",
    )
    detect_test_stream.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera name (from Frigate config)",
    )
    detect_test_stream.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)",
    )
    detect_test_stream.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frames per second (default: 5.0)",
    )
    detect_test_stream.add_argument(
        "--save-frame",
        type=str,
        help="Save a sample frame to this path",
    )

    # detect cameras: List available cameras
    detect_cameras = detect_subparsers.add_parser(
        "cameras",
        help="List available cameras from Frigate",
    )

    # detect test-frame: Test cat detection on an image
    detect_test_frame = detect_subparsers.add_parser(
        "test-frame",
        help="Test cat detection on a single image",
    )
    detect_test_frame.add_argument(
        "image_path",
        type=str,
        help="Path to image file",
    )
    detect_test_frame.add_argument(
        "--output", "-o",
        type=str,
        help="Save annotated image to this path",
    )
    detect_test_frame.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    detect_test_frame.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    detect_test_frame.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)",
    )

    # detect test-tracker: Test tracking on live stream
    detect_test_tracker = detect_subparsers.add_parser(
        "test-tracker",
        help="Test cat tracking on a live camera stream",
    )
    detect_test_tracker.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera name (from Frigate config)",
    )
    detect_test_tracker.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)",
    )
    detect_test_tracker.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frames per second (default: 5.0)",
    )
    detect_test_tracker.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    detect_test_tracker.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Minimum detection confidence (default: 0.1)",
    )
    detect_test_tracker.add_argument(
        "--max-disappeared",
        type=int,
        default=30,
        help="Frames before track is lost (default: 30)",
    )
    detect_test_tracker.add_argument(
        "--max-distance",
        type=float,
        default=0.2,
        help="Max centroid distance for matching (default: 0.2)",
    )
    detect_test_tracker.add_argument(
        "--output-dir",
        type=str,
        help="Save annotated frames to this directory",
    )
    detect_test_tracker.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)",
    )

    # detect test-events: Test event management with live stream
    detect_test_events = detect_subparsers.add_parser(
        "test-events",
        help="Test event management on a live camera stream",
    )
    detect_test_events.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera name (from Frigate config)",
    )
    detect_test_events.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
    )
    detect_test_events.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frames per second (default: 5.0)",
    )
    detect_test_events.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    detect_test_events.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Minimum detection confidence (default: 0.1)",
    )
    detect_test_events.add_argument(
        "--min-event-duration",
        type=float,
        default=0.5,
        help="Minimum event duration in seconds (default: 0.5)",
    )
    detect_test_events.add_argument(
        "--disappeared-timeout",
        type=float,
        default=2.0,
        help="Seconds before ending event after cat disappears (default: 2.0)",
    )
    detect_test_events.add_argument(
        "--output-dir",
        type=str,
        help="Save best frames from events to this directory",
    )
    detect_test_events.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)",
    )

    # detect run: Run detection pipeline on a single camera
    detect_run = detect_subparsers.add_parser(
        "run",
        help="Run detection pipeline on a single camera",
    )
    detect_run.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera name (from Frigate config)",
    )
    detect_run.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run duration in seconds (0 = run forever, default: 0)",
    )
    detect_run.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frames per second (default: 5.0)",
    )
    detect_run.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    detect_run.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Minimum detection confidence (default: 0.1)",
    )
    detect_run.add_argument(
        "--output-dir",
        type=str,
        help="Save event frames to this directory (default: data/detection/events)",
    )
    detect_run.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cuda, cpu (default: auto)",
    )

    # detect start: Start detection service with multiple cameras
    detect_start = detect_subparsers.add_parser(
        "start",
        help="Start detection service with multiple cameras",
    )
    detect_start.add_argument(
        "--cameras",
        type=str,
        help="Comma-separated list of cameras (default: from config or all enabled)",
    )
    detect_start.add_argument(
        "--fps",
        type=float,
        help="Target frames per second (default: from config)",
    )
    detect_start.add_argument(
        "--model",
        type=str,
        help="YOLO model to use (default: from config)",
    )
    detect_start.add_argument(
        "--confidence",
        type=float,
        help="Minimum detection confidence (default: from config)",
    )
    detect_start.add_argument(
        "--output-dir",
        type=str,
        help="Save event frames to this directory (default: from config)",
    )
    detect_start.add_argument(
        "--device",
        type=str,
        help="Device for inference: auto, cuda, cpu (default: from config)",
    )

    # detect status: Show detection configuration and check readiness
    detect_status = detect_subparsers.add_parser(
        "status",
        help="Show detection configuration and check service readiness",
    )

    # detect import-events: Import saved detection events into labeling database
    detect_import = detect_subparsers.add_parser(
        "import-events",
        help="Import saved detection events into the labeling database",
    )
    detect_import.add_argument(
        "--events-dir",
        type=str,
        help="Directory containing saved detection events (default: data/detection/events)",
    )
    detect_import.add_argument(
        "--db",
        type=str,
        help="Path to labeling database (default: data/training/samples.db)",
    )
    detect_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without making changes",
    )

    args = parser.parse_args()

    if args.command == "label":
        from cat_watcher.labeling.app import main as label_main

        sys.argv = ["cat-watcher-label"]
        sys.argv.extend(["--host", args.host])
        sys.argv.extend(["--port", str(args.port)])
        if args.reload:
            sys.argv.append("--reload")
        label_main()

    elif args.command == "web":
        from cat_watcher.web.app import main as web_main

        sys.argv = ["cat-watcher-web"]
        sys.argv.extend(["--host", args.host])
        sys.argv.extend(["--port", str(args.port)])
        if args.reload:
            sys.argv.append("--reload")
        web_main()

    elif args.command == "train":
        if args.train_type == "behavior":
            from cat_watcher.training.behavior import train_behavior_model

            print("Training behavior detection model...")
            print(f"  Data: {args.data_dir}")
            print(f"  Output: {args.output_dir}")
            print(f"  Epochs: {args.epochs}")
            print(f"  Model: YOLOv8{args.model_size}")

            results = train_behavior_model(
                data_dir=Path(args.data_dir),
                output_dir=Path(args.output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_size=args.model_size,
                device=args.device,
                export_onnx=not args.no_export,
            )

            print("\nTraining complete!")
            print(f"  Best model: {results['best_model']}")
            if "exported" in results:
                print(f"  ONNX model: {results['exported'].get('onnx', 'N/A')}")
            if "metrics" in results:
                print(f"  mAP50: {results['metrics'].get('mAP50', 0):.4f}")
                print(f"  mAP50-95: {results['metrics'].get('mAP50-95', 0):.4f}")

        elif args.train_type == "catid":
            from cat_watcher.training.cat_id import train_cat_id_model

            print("Training cat identification model...")
            print(f"  Data: {args.data_dir}")
            print(f"  Output: {args.output_dir}")
            print(f"  Epochs: {args.epochs}")
            print(f"  Model: {args.model}")

            results = train_cat_id_model(
                data_dir=Path(args.data_dir),
                output_dir=Path(args.output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_name=args.model,
                device=args.device,
                export_onnx=not args.no_export,
            )

            print("\nTraining complete!")
            print(f"  Best model: {results['best_model']}")
            print(f"  Best accuracy: {results['best_accuracy']:.4f}")
            if "onnx_model" in results:
                print(f"  ONNX model: {results['onnx_model']}")

        else:
            train_parser.print_help()
            return 1

    elif args.command == "prepare":
        if args.prepare_type == "behavior":
            from cat_watcher.training.dataset import split_dataset

            print("Preparing behavior training data...")
            print(f"  Source DB: {args.db}")
            print(f"  Output: {args.output_dir}")
            print(f"  Val ratio: {args.val_ratio}")

            stats = split_dataset(
                storage_db=Path(args.db),
                output_dir=Path(args.output_dir),
                val_ratio=args.val_ratio,
            )

            print("\nData preparation complete!")
            print(f"  Training samples: {stats['train']}")
            print(f"  Validation samples: {stats['val']}")
            print(f"  Total: {stats['total']}")

        elif args.prepare_type == "catid":
            from cat_watcher.training.dataset import split_cat_id_dataset

            print("Preparing cat ID training data...")
            print(f"  Source DB: {args.db}")
            print(f"  Output: {args.output_dir}")
            print(f"  Val ratio: {args.val_ratio}")

            stats = split_cat_id_dataset(
                storage_db=Path(args.db),
                output_dir=Path(args.output_dir),
                val_ratio=args.val_ratio,
            )

            print("\nData preparation complete!")
            print(f"  Training samples: {stats['train']}")
            print(f"  Validation samples: {stats['val']}")
            print(f"  Total: {stats['total']}")

        else:
            prepare_parser.print_help()
            return 1

    elif args.command == "inference":
        if args.inference_type == "api":
            import uvicorn  # noqa: E402

            from cat_watcher.inference.app import create_app  # noqa: E402

            print("Starting inference API server...")
            print(f"  Host: {args.host}")
            print(f"  Port: {args.port}")
            print(f"  Behavior model: {args.behavior_model or 'None'}")
            print(f"  Cat ID model: {args.catid_model or 'None'}")

            app = create_app(
                behavior_model=args.behavior_model,
                catid_model=args.catid_model,
                use_onnx=not args.no_onnx,
                device=args.device,
            )

            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                reload=args.reload,
            )

        elif args.inference_type == "image":
            import json

            from cat_watcher.inference.pipeline import InferencePipeline, PipelineConfig

            print(f"Running inference on: {args.image_path}")

            config = PipelineConfig(
                behavior_model_path=args.behavior_model or "",
                catid_model_path=args.catid_model or "",
                use_onnx=not args.no_onnx,
                device=args.device,
            )

            pipeline = InferencePipeline(config)
            pipeline.load()

            result = pipeline.process(args.image_path, source="cli")

            print("\nResults:")
            print(json.dumps(result.to_dict(), indent=2))

            if args.output and result.detections:
                # Save annotated image
                from PIL import Image, ImageDraw

                img = Image.open(args.image_path)
                draw = ImageDraw.Draw(img)
                img_w, img_h = img.size

                for i, detection in enumerate(result.detections):
                    bbox = detection.bbox
                    # Convert normalized coords to pixels
                    x1 = int(bbox.x_min * img_w)
                    y1 = int(bbox.y_min * img_h)
                    x2 = int(bbox.x_max * img_w)
                    y2 = int(bbox.y_max * img_h)
                    # Draw box
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline="green",
                        width=3,
                    )
                    # Draw label
                    label = f"{detection.behavior.value} ({detection.confidence:.2f})"
                    if i < len(result.identifications):
                        label += f" - {result.identifications[i].cat.value}"
                    draw.text((bbox.x, bbox.y - 20), label, fill="green")

                img.save(args.output)
                print(f"\nAnnotated image saved to: {args.output}")

        else:
            inference_parser.print_help()
            return 1

    elif args.command in ("homeassistant", "ha"):
        import asyncio

        from cat_watcher.config import get_settings
        from cat_watcher.mqtt import MQTTPublisher
        from cat_watcher.homeassistant import HAEventPublisher
        from cat_watcher.schemas import BehaviorType, CatName

        settings = get_settings()

        async def run_ha_command() -> int:
            mqtt = MQTTPublisher(
                broker=settings.mqtt.broker,
                port=settings.mqtt.port,
                username=settings.mqtt.username,
                password=settings.mqtt.password,
                topic_prefix=getattr(args, "topic_prefix", "cat_watcher"),
            )
            await mqtt.connect()

            try:
                publisher = HAEventPublisher(
                    mqtt_publisher=mqtt,
                    topic_prefix=getattr(args, "topic_prefix", "cat_watcher"),
                    device_name=getattr(args, "device_name", "Cat Watcher"),
                )

                if args.ha_type == "discover":
                    if args.remove:
                        print("Removing Home Assistant discovery messages...")
                        await publisher.remove_discovery()
                        print("Done! Entities should be removed from Home Assistant.")
                    else:
                        print("Publishing Home Assistant discovery messages...")
                        await publisher.publish_discovery()
                        print("Done! Check Home Assistant for new entities.")

                elif args.ha_type == "status":
                    online = not getattr(args, "offline", False)
                    status = "online" if online else "offline"
                    print(f"Publishing status: {status}")
                    await publisher.publish_status(online=online)
                    print("Done!")

                elif args.ha_type == "test":
                    behavior_value = args.behavior
                    cat_value = args.cat

                    # Parse behavior
                    try:
                        behavior = BehaviorType(behavior_value)
                    except ValueError:
                        print(f"Unknown behavior: {behavior_value}")
                        print(f"Valid behaviors: {[b.value for b in BehaviorType]}")
                        return 1

                    # Parse cat
                    try:
                        cat = CatName(cat_value)
                    except ValueError:
                        print(f"Unknown cat: {cat_value}")
                        print(f"Valid cats: {[c.value for c in CatName]}")
                        return 1

                    print("Publishing test detection:")
                    print(f"  Behavior: {behavior.value}")
                    print(f"  Cat: {cat.value}")
                    print(f"  Confidence: {args.confidence}")

                    await publisher.publish_detection(
                        behavior=behavior,
                        cat=cat,
                        confidence=args.confidence,
                        camera="test",
                        event_id="test-" + str(int(asyncio.get_event_loop().time())),
                    )
                    print("Done! Check Home Assistant for the event.")

                else:
                    ha_parser.print_help()
                    return 1

                return 0

            finally:
                await mqtt.disconnect()

        return asyncio.run(run_ha_command())

    elif args.command == "detect":
        import asyncio

        from cat_watcher.config import get_settings

        settings = get_settings()

        if args.detect_type == "test-stream":
            from cat_watcher.detection.stream import (
                StreamReader,
                get_camera_rtsp_url,
            )

            async def run_test_stream() -> int:
                camera = args.camera
                duration = args.duration
                target_fps = args.fps
                save_frame = args.save_frame

                print(f"Testing stream for camera: {camera}")
                print(f"Duration: {duration}s, Target FPS: {target_fps}")

                # Get RTSP URL from Frigate
                try:
                    rtsp_url = await get_camera_rtsp_url(
                        settings.frigate.url,
                        camera,
                        rtsp_username=settings.frigate.rtsp_username,
                        rtsp_password=settings.frigate.rtsp_password,
                    )
                    print(f"RTSP URL: {rtsp_url.split('@')[-1] if '@' in rtsp_url else rtsp_url}")
                except Exception as e:
                    print(f"Error getting camera URL: {e}")
                    return 1

                # Create stream reader
                reader = StreamReader(
                    url=rtsp_url,
                    target_fps=target_fps,
                )

                try:
                    print("Connecting to stream...")
                    await reader.start()
                    print(f"Connected! Resolution: {reader.resolution[0]}x{reader.resolution[1]}")
                    print(f"Source FPS: {reader.source_fps}")
                    print()

                    # Read frames for duration
                    import time
                    start_time = time.time()
                    frame_count = 0
                    saved_frame = False

                    async for timestamp, frame in reader.frames():
                        frame_count += 1
                        elapsed = time.time() - start_time

                        # Save first frame if requested
                        if save_frame and not saved_frame:
                            import cv2
                            cv2.imwrite(save_frame, frame)
                            print(f"Saved frame to: {save_frame}")
                            saved_frame = True

                        # Print progress every second
                        if frame_count % int(target_fps) == 0:
                            stats = reader.stats
                            print(
                                f"[{elapsed:.1f}s] Frames: {frame_count}, "
                                f"FPS: {stats.effective_fps:.1f}, "
                                f"Dropped: {stats.frames_dropped}"
                            )

                        if elapsed >= duration:
                            break

                    print()
                    print("Test complete!")
                    stats = reader.stats
                    print(f"  Total frames yielded: {stats.frames_yielded}")
                    print(f"  Total frames read: {stats.frames_read}")
                    print(f"  Frames dropped: {stats.frames_dropped}")
                    print(f"  Effective FPS: {stats.effective_fps:.2f}")
                    print(f"  Reconnects: {stats.reconnects}")
                    print(f"  Errors: {stats.errors}")

                    return 0

                except ConnectionError as e:
                    print(f"Connection error: {e}")
                    return 1
                except Exception as e:
                    print(f"Error: {e}")
                    return 1
                finally:
                    await reader.stop()

            return asyncio.run(run_test_stream())

        elif args.detect_type == "cameras":
            from cat_watcher.detection.stream import list_cameras

            async def run_list_cameras() -> int:
                try:
                    cameras = await list_cameras(settings.frigate.url)
                    print(f"Available cameras from {settings.frigate.url}:")
                    print()
                    for cam in cameras:
                        status = "enabled" if cam["enabled"] else "disabled"
                        print(
                            f"  {cam['name']:20} "
                            f"{cam['width']}x{cam['height']} @ {cam['fps']}fps "
                            f"({status})"
                        )
                    return 0
                except Exception as e:
                    print(f"Error: {e}")
                    return 1

            return asyncio.run(run_list_cameras())

        elif args.detect_type == "status":
            from cat_watcher.detection.stream import list_cameras

            async def run_status() -> int:
                det_config = settings.detection

                print("=" * 60)
                print("Cat Watcher Detection Configuration")
                print("=" * 60)
                print()

                # Detection settings
                print("Detection Settings:")
                print(f"  Cat model:        {det_config.cat_model}")
                print(f"  Cat confidence:   {det_config.cat_confidence}")
                print(f"  Behavior model:   {det_config.behavior_model or 'None'}")
                print(f"  Device:           {det_config.device}")
                print(f"  Frame rate:       {det_config.frame_rate} FPS")
                print()

                # Event settings
                print("Event Settings:")
                print(f"  Min duration:     {det_config.min_event_duration}s")
                print(f"  Max duration:     {det_config.max_event_duration}s")
                print(f"  Cooldown:         {det_config.event_cooldown}s")
                print(f"  Disappear timeout:{det_config.disappeared_timeout}s")
                print()

                # Output settings
                print("Output Settings:")
                print(f"  Output dir:       {det_config.output_dir}")
                print(f"  Save frames:      {det_config.save_frames}")
                print(f"  DB path:          {det_config.db_path}")
                print()

                # Frigate connection
                print("Frigate Connection:")
                print(f"  URL:              {settings.frigate.url}")
                print(f"  RTSP credentials: {'configured' if settings.frigate.rtsp_username else 'not set'}")
                print()

                # Available cameras
                print("Available Cameras:")
                try:
                    cameras = await list_cameras(settings.frigate.url)
                    configured_cameras = set(det_config.cameras) if det_config.cameras else None

                    for cam in cameras:
                        status = "enabled" if cam["enabled"] else "disabled"
                        selected = ""
                        if configured_cameras:
                            selected = " [SELECTED]" if cam["name"] in configured_cameras else ""
                        elif cam["enabled"]:
                            selected = " [SELECTED]"

                        print(
                            f"  {cam['name']:20} "
                            f"{cam['width']}x{cam['height']} @ {cam['fps']}fps "
                            f"({status}){selected}"
                        )

                    print()
                    if configured_cameras:
                        print(f"  Configured cameras: {', '.join(det_config.cameras)}")
                    else:
                        print("  Configured cameras: all enabled (default)")

                except Exception as e:
                    print(f"  Error fetching cameras: {e}")

                print()
                print("=" * 60)
                print("Run 'cat-watcher detect start' to begin detection")
                print("=" * 60)

                return 0

            return asyncio.run(run_status())

        elif args.detect_type == "test-frame":
            from pathlib import Path

            from cat_watcher.detection.cat_detector import CatDetector

            image_path = Path(args.image_path)
            if not image_path.exists():
                print(f"Error: Image not found: {image_path}")
                return 1

            print(f"Testing cat detection on: {image_path}")
            print(f"Model: {args.model}")
            print(f"Confidence threshold: {args.confidence}")
            print(f"Device: {args.device}")
            print()

            try:
                detector = CatDetector(
                    model_path=args.model,
                    confidence_threshold=args.confidence,
                    device=args.device,
                )

                if args.output:
                    detections, _ = detector.detect_and_annotate(
                        image_path,
                        output_path=args.output,
                    )
                    print(f"Annotated image saved to: {args.output}")
                else:
                    detections = detector.detect(image_path)

                print()
                if detections:
                    print(f"Found {len(detections)} cat(s):")
                    for i, det in enumerate(detections, 1):
                        print(
                            f"  {i}. confidence={det.confidence:.3f}, "
                            f"bbox=[{det.bbox.x_min:.3f}, {det.bbox.y_min:.3f}, "
                            f"{det.bbox.x_max:.3f}, {det.bbox.y_max:.3f}], "
                            f"pixels={det.bbox_pixels}"
                        )
                else:
                    print("No cats detected.")

                return 0

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                return 1

        elif args.detect_type == "test-tracker":
            import time
            from pathlib import Path

            from cat_watcher.detection.cat_detector import CatDetector
            from cat_watcher.detection.stream import (
                StreamReader,
                get_camera_rtsp_url,
            )
            from cat_watcher.detection.tracker import CentroidTracker

            async def run_test_tracker() -> int:
                camera = args.camera
                duration = args.duration
                target_fps = args.fps
                output_dir = Path(args.output_dir) if args.output_dir else None

                print(f"Testing tracker for camera: {camera}")
                print(f"Duration: {duration}s, Target FPS: {target_fps}")
                print(f"Detection confidence: {args.confidence}")
                print(f"Max disappeared: {args.max_disappeared} frames")
                print(f"Max distance: {args.max_distance}")
                print()

                # Create output directory if needed
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Saving annotated frames to: {output_dir}")
                    print()

                # Get RTSP URL from Frigate
                try:
                    rtsp_url = await get_camera_rtsp_url(
                        settings.frigate.url,
                        camera,
                        rtsp_username=settings.frigate.rtsp_username,
                        rtsp_password=settings.frigate.rtsp_password,
                    )
                except Exception as e:
                    print(f"Error getting camera URL: {e}")
                    return 1

                # Create components
                reader = StreamReader(url=rtsp_url, target_fps=target_fps)
                detector = CatDetector(
                    model_path=args.model,
                    confidence_threshold=args.confidence,
                    device=args.device,
                )
                tracker = CentroidTracker(
                    max_disappeared=args.max_disappeared,
                    max_distance=args.max_distance,
                )

                try:
                    print("Connecting to stream...")
                    await reader.start()
                    print(f"Connected! Resolution: {reader.resolution[0]}x{reader.resolution[1]}")
                    print()

                    start_time = time.time()
                    frame_count = 0
                    total_detections = 0
                    frames_with_cats = 0

                    async for timestamp, frame in reader.frames():
                        frame_count += 1
                        elapsed = time.time() - start_time

                        # Detect cats in frame
                        detections = detector.detect(frame)
                        total_detections += len(detections)
                        if detections:
                            frames_with_cats += 1

                        # Update tracker
                        tracked = tracker.update(detections)

                        # Print status on detection
                        if tracked:
                            track_info = ", ".join(
                                f"T{t.track_id}({t.confidence:.2f})"
                                for t in tracked
                            )
                            print(
                                f"[{elapsed:.1f}s] Frame {frame_count}: "
                                f"Tracking {len(tracked)} cat(s): {track_info}"
                            )

                            # Save annotated frame if requested
                            if output_dir:
                                import cv2
                                annotated = frame.copy()
                                h, w = frame.shape[:2]

                                for t in tracked:
                                    # Draw bounding box
                                    x1 = int(t.bbox.x_min * w)
                                    y1 = int(t.bbox.y_min * h)
                                    x2 = int(t.bbox.x_max * w)
                                    y2 = int(t.bbox.y_max * h)
                                    cv2.rectangle(
                                        annotated,
                                        (x1, y1),
                                        (x2, y2),
                                        (0, 255, 0),
                                        2,
                                    )
                                    # Draw label
                                    label = f"T{t.track_id} ({t.confidence:.2f})"
                                    cv2.putText(
                                        annotated,
                                        label,
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),
                                        2,
                                    )

                                frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                                cv2.imwrite(str(frame_path), annotated)

                        # Check duration
                        if elapsed >= duration:
                            break

                    print()
                    print("=" * 50)
                    print("Test complete!")
                    print()
                    print("Stream stats:")
                    stats = reader.stats
                    print(f"  Total frames: {frame_count}")
                    print(f"  Effective FPS: {stats.effective_fps:.2f}")
                    print()
                    print("Detection stats:")
                    print(f"  Total detections: {total_detections}")
                    print(f"  Frames with cats: {frames_with_cats}")
                    if frame_count > 0:
                        print(f"  Cat presence: {100*frames_with_cats/frame_count:.1f}%")
                    print()
                    print("Tracker stats:")
                    tracker_stats = tracker.stats
                    print(f"  Total tracks created: {tracker_stats['total_tracks_created']}")
                    print(f"  Active tracks: {tracker_stats['active_tracks']}")
                    print(f"  Still tracked: {tracker_stats['total_tracks']}")

                    # Show final track summaries
                    all_tracks = tracker.get_all_tracks()
                    if all_tracks:
                        print()
                        print("Track summaries:")
                        for t in all_tracks:
                            print(
                                f"  Track {t.track_id}: "
                                f"frames={t.total_frames}, "
                                f"max_conf={t.max_confidence:.2f}, "
                                f"missing={t.frames_since_seen}"
                            )

                    return 0

                except ConnectionError as e:
                    print(f"Connection error: {e}")
                    return 1
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
                finally:
                    await reader.stop()

            return asyncio.run(run_test_tracker())

        elif args.detect_type == "test-events":
            import time
            from pathlib import Path

            from cat_watcher.detection.cat_detector import CatDetector
            from cat_watcher.detection.event_manager import CatEvent, EventManager
            from cat_watcher.detection.stream import (
                StreamReader,
                get_camera_rtsp_url,
            )
            from cat_watcher.detection.tracker import CentroidTracker

            async def run_test_events() -> int:
                camera = args.camera
                duration = args.duration
                target_fps = args.fps
                output_dir = Path(args.output_dir) if args.output_dir else None

                print(f"Testing event management for camera: {camera}")
                print(f"Duration: {duration}s, Target FPS: {target_fps}")
                print(f"Detection confidence: {args.confidence}")
                print(f"Min event duration: {args.min_event_duration}s")
                print(f"Disappeared timeout: {args.disappeared_timeout}s")
                print()

                # Create output directory if needed
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Saving event frames to: {output_dir}")
                    print()

                # Track completed events for summary
                completed_events: list[CatEvent] = []

                async def on_event_start(event: CatEvent) -> None:
                    print(f"[EVENT START] {event.id[:20]}... track={event.track_id}")

                async def on_event_end(event: CatEvent) -> None:
                    print(
                        f"[EVENT END] {event.id[:20]}... "
                        f"duration={event.duration:.1f}s, "
                        f"best_conf={event.best_confidence:.2f}"
                    )
                    completed_events.append(event)

                    # Save best frame if output directory specified
                    if output_dir and event.best_frame is not None:
                        import cv2
                        frame_path = output_dir / f"{event.id}.jpg"
                        
                        # Draw bbox on frame
                        annotated = event.best_frame.copy()
                        h, w = annotated.shape[:2]
                        if event.best_bbox:
                            x1 = int(event.best_bbox.x_min * w)
                            y1 = int(event.best_bbox.y_min * h)
                            x2 = int(event.best_bbox.x_max * w)
                            y2 = int(event.best_bbox.y_max * h)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"conf={event.best_confidence:.2f}"
                            cv2.putText(
                                annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
                        
                        cv2.imwrite(str(frame_path), annotated)
                        print(f"  Saved: {frame_path.name}")

                # Get RTSP URL from Frigate
                try:
                    rtsp_url = await get_camera_rtsp_url(
                        settings.frigate.url,
                        camera,
                        rtsp_username=settings.frigate.rtsp_username,
                        rtsp_password=settings.frigate.rtsp_password,
                    )
                except Exception as e:
                    print(f"Error getting camera URL: {e}")
                    return 1

                # Create components
                reader = StreamReader(url=rtsp_url, target_fps=target_fps)
                detector = CatDetector(
                    model_path=args.model,
                    confidence_threshold=args.confidence,
                    device=args.device,
                )
                tracker = CentroidTracker(
                    max_disappeared=int(target_fps * args.disappeared_timeout),
                    max_distance=0.2,
                )
                event_manager = EventManager(
                    min_event_duration=args.min_event_duration,
                    disappeared_timeout=args.disappeared_timeout,
                    on_event_start=on_event_start,
                    on_event_end=on_event_end,
                )

                try:
                    print("Connecting to stream...")
                    await reader.start()
                    print(f"Connected! Resolution: {reader.resolution[0]}x{reader.resolution[1]}")
                    print()
                    print("Watching for cats...")
                    print("-" * 50)

                    start_time = time.time()
                    frame_count = 0

                    async for timestamp, frame in reader.frames():
                        frame_count += 1
                        elapsed = time.time() - start_time

                        # Detect cats
                        detections = detector.detect(frame)

                        # Update tracker
                        tracked = tracker.update(detections)

                        # Process events
                        await event_manager.process_frame(
                            camera=camera,
                            timestamp=timestamp,
                            frame=frame,
                            tracked_objects=tracked,
                        )

                        # Check duration
                        if elapsed >= duration:
                            break

                    # End any remaining events
                    print()
                    print("-" * 50)
                    print("Ending remaining events...")
                    await event_manager.end_all_events(reason="test_complete")

                    print()
                    print("=" * 50)
                    print("Test complete!")
                    print()
                    print("Event Manager stats:")
                    stats = event_manager.stats
                    print(f"  Events created: {stats['total_events_created']}")
                    print(f"  Events completed: {stats['total_events_completed']}")
                    print(f"  Events discarded: {stats['total_events_discarded']}")
                    print()
                    print("Stream stats:")
                    print(f"  Frames processed: {frame_count}")
                    stream_stats = reader.stats
                    print(f"  Effective FPS: {stream_stats.effective_fps:.2f}")

                    if completed_events:
                        print()
                        print("Completed events:")
                        for event in completed_events:
                            print(
                                f"  - {event.id[:30]}... "
                                f"duration={event.duration:.1f}s, "
                                f"best_conf={event.best_confidence:.2f}"
                            )

                    return 0

                except ConnectionError as e:
                    print(f"Connection error: {e}")
                    return 1
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
                finally:
                    await reader.stop()

            return asyncio.run(run_test_events())

        elif args.detect_type == "run":
            import signal
            import time
            from pathlib import Path

            from cat_watcher.detection.cat_detector import CatDetector
            from cat_watcher.detection.event_manager import CatEvent
            from cat_watcher.detection.pipeline import (
                DetectionPipeline,
                PipelineSettings,
            )
            from cat_watcher.detection.stream import get_camera_rtsp_url

            async def run_pipeline() -> int:
                camera = args.camera
                duration = args.duration
                output_dir = Path(args.output_dir) if args.output_dir else Path("data/detection/events")

                print(f"Starting detection pipeline for camera: {camera}")
                print(f"Duration: {'forever' if duration == 0 else f'{duration}s'}")
                print(f"Target FPS: {args.fps}")
                print(f"Detection confidence: {args.confidence}")
                print(f"Output directory: {output_dir}")
                print()

                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Get RTSP URL
                try:
                    rtsp_url = await get_camera_rtsp_url(
                        settings.frigate.url,
                        camera,
                        rtsp_username=settings.frigate.rtsp_username,
                        rtsp_password=settings.frigate.rtsp_password,
                    )
                except Exception as e:
                    print(f"Error getting camera URL: {e}")
                    return 1

                # Create detector
                print("Loading cat detector...")
                detector = CatDetector(
                    model_path=args.model,
                    confidence_threshold=args.confidence,
                    device=args.device,
                )

                # Event callback
                events_saved = 0

                async def on_event(event: CatEvent) -> None:
                    nonlocal events_saved
                    
                    print(
                        f"[EVENT] {event.id[:30]}... "
                        f"duration={event.duration:.1f}s, "
                        f"confidence={event.best_confidence:.2f}"
                    )

                    # Save frame
                    if event.best_frame is not None:
                        import cv2
                        frame_path = output_dir / f"{event.id}.jpg"
                        
                        # Draw bbox on frame
                        annotated = event.best_frame.copy()
                        h, w = annotated.shape[:2]
                        if event.best_bbox:
                            x1 = int(event.best_bbox.x_min * w)
                            y1 = int(event.best_bbox.y_min * h)
                            x2 = int(event.best_bbox.x_max * w)
                            y2 = int(event.best_bbox.y_max * h)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        cv2.imwrite(str(frame_path), annotated)
                        events_saved += 1
                        print(f"  Saved: {frame_path.name}")

                    # Save to database (always enabled)
                    if event.best_frame is not None and event.best_bbox:
                        try:
                            from cat_watcher.collection.storage import FrameStorage
                            
                            storage = FrameStorage(Path("data/training"))
                            storage.save_detection_sample(
                                event_id=event.id,
                                camera=event.camera,
                                timestamp=event.best_timestamp.timestamp(),
                                frame=event.best_frame,
                                bbox=event.best_bbox,
                                confidence=event.best_confidence,
                                track_id=event.track_id,
                            )
                            print(f"  Saved to database")
                        except Exception as e:
                            print(f"  Error saving to database: {e}")

                # Create pipeline
                pipeline_settings = PipelineSettings(
                    target_fps=args.fps,
                    confidence_threshold=args.confidence,
                )

                pipeline = DetectionPipeline(
                    camera=camera,
                    stream_url=rtsp_url,
                    cat_detector=detector,
                    event_callback=on_event,
                    settings=pipeline_settings,
                )

                # Handle Ctrl+C gracefully
                stop_requested = False

                def signal_handler(sig, frame):
                    nonlocal stop_requested
                    if not stop_requested:
                        print("\nShutdown requested, stopping pipeline...")
                        stop_requested = True

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                try:
                    print("Starting pipeline...")
                    await pipeline.start()
                    print(f"Pipeline running. Press Ctrl+C to stop.")
                    print("-" * 50)

                    start_time = time.time()
                    
                    # Run until duration or Ctrl+C
                    while not stop_requested:
                        await asyncio.sleep(1.0)
                        
                        # Print status every 30 seconds
                        elapsed = time.time() - start_time
                        if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                            status = pipeline.get_status()
                            print(
                                f"[{elapsed:.0f}s] "
                                f"frames={status['stats']['frames_processed']}, "
                                f"detections={status['stats']['detections_total']}, "
                                f"events={status['stats']['events_completed']}"
                            )
                        
                        # Check duration
                        if duration > 0 and elapsed >= duration:
                            print(f"\nDuration reached ({duration}s)")
                            break

                    print()
                    print("-" * 50)
                    print("Stopping pipeline...")
                    await pipeline.stop()

                    # Print summary
                    print()
                    print("=" * 50)
                    print("Pipeline Summary")
                    print("=" * 50)
                    status = pipeline.get_status()
                    stats = status["stats"]
                    print(f"Runtime: {stats['runtime']:.1f}s")
                    print(f"Frames processed: {stats['frames_processed']}")
                    print(f"Effective FPS: {stats['effective_fps']:.2f}")
                    print(f"Total detections: {stats['detections_total']}")
                    print(f"Events started: {stats['events_started']}")
                    print(f"Events completed: {stats['events_completed']}")
                    print(f"Events discarded: {stats['events_discarded']}")
                    print(f"Errors: {stats['errors']}")
                    print(f"Frames saved: {events_saved}")

                    return 0

                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
                finally:
                    if pipeline.is_running:
                        await pipeline.stop()

            return asyncio.run(run_pipeline())

        elif args.detect_type == "start":
            import signal
            import time
            from pathlib import Path

            from cat_watcher.detection.service import (
                DetectionService,
                ServiceSettings,
            )

            async def run_service() -> int:
                # Get detection config
                det_config = settings.detection

                # Use CLI args if provided, otherwise use config
                cameras_arg = args.cameras.split(",") if args.cameras else None
                cameras = cameras_arg if cameras_arg else (det_config.cameras or None)
                fps = args.fps if args.fps is not None else det_config.frame_rate
                model = args.model if args.model is not None else det_config.cat_model
                confidence = args.confidence if args.confidence is not None else det_config.cat_confidence
                device = args.device if args.device is not None else det_config.device
                output_dir = Path(args.output_dir) if args.output_dir is not None else det_config.output_dir

                print("=" * 60)
                print("Cat Watcher Detection Service")
                print("=" * 60)
                print()
                print(f"Frigate URL: {settings.frigate.url}")
                print(f"Cameras: {cameras if cameras else 'all enabled'}")
                print(f"Target FPS: {fps}")
                print(f"Model: {model}")
                print(f"Confidence: {confidence}")
                print(f"Device: {device}")
                print(f"Output dir: {output_dir}")
                print(f"DB path: {det_config.db_path}")
                print()

                # Create service settings from config + overrides
                service_settings = ServiceSettings(
                    cat_model=model,
                    cat_confidence=confidence,
                    device=device,
                    target_fps=fps,
                    min_event_duration=det_config.min_event_duration,
                    max_event_duration=det_config.max_event_duration,
                    event_cooldown=det_config.event_cooldown,
                    disappeared_timeout=det_config.disappeared_timeout,
                    output_dir=output_dir,
                    save_frames=det_config.save_frames,
                    db_path=det_config.db_path,
                )

                # Create service
                service = DetectionService(
                    frigate_url=settings.frigate.url,
                    settings=service_settings,
                    rtsp_username=settings.frigate.rtsp_username,
                    rtsp_password=settings.frigate.rtsp_password,
                )

                # Handle Ctrl+C
                stop_requested = False

                def signal_handler(sig, frame):
                    nonlocal stop_requested
                    if not stop_requested:
                        print("\nShutdown requested...")
                        stop_requested = True

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                try:
                    print("Starting service...")
                    await service.start(cameras=cameras)

                    print()
                    print("-" * 60)
                    print("Service running. Press Ctrl+C to stop.")
                    print("-" * 60)
                    print()

                    # Run until stopped
                    last_status_time = 0
                    while not stop_requested:
                        await asyncio.sleep(1.0)

                        # Print status every 60 seconds
                        now = time.time()
                        if now - last_status_time >= 60:
                            status = service.get_status()
                            print(
                                f"[{status['runtime']:.0f}s] "
                                f"cameras={len(status['cameras'])}, "
                                f"frames={status['summary']['total_frames']}, "
                                f"events={status['summary']['total_events']}"
                            )
                            last_status_time = now

                    print()
                    print("-" * 60)
                    print("Stopping service...")
                    await service.stop()

                    # Print final status
                    print()
                    print("=" * 60)
                    print("Service Summary")
                    print("=" * 60)
                    status = service.get_status()
                    print(f"Runtime: {status['runtime']:.1f}s")
                    print()
                    print("Per-camera stats:")
                    for cam_name, cam_status in status["cameras"].items():
                        print(
                            f"  {cam_name}: "
                            f"frames={cam_status['frames']}, "
                            f"detections={cam_status['detections']}, "
                            f"events={cam_status['events']}"
                        )
                    print()
                    print("Totals:")
                    summary = status["summary"]
                    print(f"  Active cameras: {summary['active_cameras']}")
                    print(f"  Total frames: {summary['total_frames']}")
                    print(f"  Total detections: {summary['total_detections']}")
                    print(f"  Total events: {summary['total_events']}")

                    return 0

                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
                finally:
                    if service.is_running:
                        await service.stop()

            return asyncio.run(run_service())

        elif args.detect_type == "import-events":
            import json
            import re
            from datetime import datetime
            from pathlib import Path

            import cv2

            from cat_watcher.collection.storage import BoundingBox, FrameStorage

            def run_import_events() -> int:
                # Get paths from args or config
                events_dir = Path(args.events_dir) if args.events_dir else Path(settings.detection.output_dir)
                db_path = Path(args.db) if args.db else settings.detection.db_path
                dry_run = args.dry_run

                print("=" * 60)
                print("Import Detection Events to Labeling Database")
                print("=" * 60)
                print()
                print(f"Events directory: {events_dir}")
                print(f"Database path:    {db_path}")
                print(f"Dry run:          {dry_run}")
                print()

                if not events_dir.exists():
                    print(f"Error: Events directory not found: {events_dir}")
                    return 1

                # Find all event JPG files
                # Expected naming: {timestamp}-{camera}-{track_id}-{uuid}.jpg
                event_files = list(events_dir.glob("*.jpg"))
                # Use set to dedupe in case of recursive glob
                seen = {f.name for f in event_files}
                for f in events_dir.glob("**/*.jpg"):
                    if f.name not in seen:
                        event_files.append(f)
                        seen.add(f.name)

                if not event_files:
                    print("No event JPG files found.")
                    return 0

                print(f"Found {len(event_files)} event files")
                print()

                # Initialize storage (skip in dry run mode to not create DB)
                storage = None
                if not dry_run:
                    storage = FrameStorage(db_path.parent)

                imported = 0
                skipped = 0
                errors = 0

                for event_file in sorted(event_files):
                    # Parse event filename
                    # Pattern: {timestamp}-{camera}-{track_id}-{uuid}.jpg
                    # Where camera can contain dashes (e.g., "apollo-dish")
                    # UUID is 8 hex chars, track_id is a number
                    filename = event_file.stem

                    try:
                        parts = filename.split("-")

                        if len(parts) >= 4 and parts[0].isdigit():
                            # New format: {timestamp}-{camera}-{track_id}-{uuid}
                            # Work backwards: uuid is last, track_id is second to last
                            timestamp_str = parts[0]
                            uuid_part = parts[-1]
                            track_id_str = parts[-2]
                            # Everything between timestamp and track_id is camera name
                            camera = "-".join(parts[1:-2])
                            
                            try:
                                track_id = int(track_id_str)
                            except ValueError:
                                track_id = 0
                            
                            # Use the full filename as event_id for uniqueness
                            event_id = filename
                        else:
                            # Unknown format - use filename as ID
                            event_id = filename
                            camera = "unknown"
                            timestamp_str = str(int(event_file.stat().st_mtime))
                            track_id = 0

                        # Parse timestamp
                        try:
                            timestamp = float(timestamp_str)
                        except ValueError:
                            timestamp = event_file.stat().st_mtime

                        # Skip if already in database
                        if storage:
                            existing = storage.get_sample(event_id)
                            if existing:
                                skipped += 1
                                continue

                        # Read frame to get dimensions and extract bbox from annotation
                        frame = cv2.imread(str(event_file))
                        if frame is None:
                            print(f"  Error reading: {event_file.name}")
                            errors += 1
                            continue

                        height, width = frame.shape[:2]

                        # Try to read metadata from associated JSON if exists
                        json_path = event_file.with_suffix(".json")
                        bbox = None
                        confidence = 0.5  # Default

                        if json_path.exists():
                            try:
                                with open(json_path) as f:
                                    meta = json.load(f)
                                    if "bbox" in meta:
                                        bbox = BoundingBox(**meta["bbox"])
                                    if "confidence" in meta:
                                        confidence = meta["confidence"]
                                    if "track_id" in meta:
                                        track_id = meta["track_id"]
                            except Exception:
                                pass

                        # Default bbox if not found
                        if bbox is None:
                            # Use center region as default
                            bbox = BoundingBox(
                                x_min=0.2,
                                y_min=0.2,
                                x_max=0.8,
                                y_max=0.8,
                            )

                        if dry_run:
                            print(f"  Would import: {event_id} ({camera}, {width}x{height})")
                        else:
                            # Save to database
                            storage.save_detection_sample(
                                event_id=event_id,
                                camera=camera,
                                timestamp=timestamp,
                                frame=frame,
                                bbox=bbox,
                                confidence=confidence,
                                track_id=track_id,
                                save_crop=True,
                            )
                            print(f"  Imported: {event_id} ({camera})")

                        imported += 1

                    except Exception as e:
                        print(f"  Error with {event_file.name}: {e}")
                        errors += 1
                        continue

                print()
                print("=" * 60)
                print("Import Summary")
                print("=" * 60)
                print(f"  Imported: {imported}")
                print(f"  Skipped (already exists): {skipped}")
                print(f"  Errors: {errors}")

                if dry_run:
                    print()
                    print("This was a dry run. Run without --dry-run to import.")

                return 0

            return run_import_events()

        else:
            detect_parser.print_help()
            return 1

    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
