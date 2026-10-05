from pathlib import Path

from voice_module import get_target_from_voice
from detector import load_image, process_frame, display_result


def main():

    print("=" * 60)
    print("       AI-POWERED OBJECT RETRIEVAL ROBOT")
    print("       VOICE + YOLO INTEGRATION TEST")
    print("=" * 60)

    # ---------------------------------------
    # STEP 1: Get target using voice
    # ---------------------------------------

    target = get_target_from_voice()

    if target is None:
        print("\n❌ Target could not be identified.")
        return

    print("\n✅ Requested target:", target)

    # ---------------------------------------
    # STEP 2: Ask for test image
    # ---------------------------------------

    image_path = input(
        "\nEnter the path of the test image: "
    ).strip().strip('"')

    image_path = Path(image_path)

    # ---------------------------------------
    # STEP 3: Load image
    # ---------------------------------------

    try:
        frame = load_image(image_path)

    except FileNotFoundError as error:
        print("\n❌", error)
        return

    # ---------------------------------------
    # STEP 4: Run YOLO
    # ---------------------------------------

    detection, result = process_frame(
        frame,
        target_object=target,
        confidence_threshold=0.5
    )

    # ---------------------------------------
    # STEP 5: Print result
    # ---------------------------------------

    print("\n" + "=" * 60)

    if detection["status"] == "TARGET_FOUND":

        print("✅ TARGET FOUND")
        print("=" * 60)

        print("Target      :", detection["target"])
        print("Confidence  :", round(
            detection["confidence"], 2
        ))
        print("Bounding Box:", detection["bbox"])
        print("Center      :", detection["center"])
        print("Position    :", detection["position"])
        print("Command     :", detection["command"])

    else:

        print("❌ TARGET NOT FOUND")
        print("=" * 60)

        print("Target      :", detection["target"])
        print("Command     :", detection["command"])

    print("=" * 60)

    # ---------------------------------------
    # STEP 6: Display annotated image
    # ---------------------------------------

    display_result(result)


if __name__ == "__main__":
    main()