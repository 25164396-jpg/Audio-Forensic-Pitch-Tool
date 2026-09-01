import os
import sys
import json
import argparse
import logging
import numpy as np
import librosa
import soundfile as sf

__version__ = "2.1.0"

# متوسط التردد الطبيعي للكلام البشري (تحديد نطاق بين 100Hz و 200Hz)
TARGET_HUMAN_PITCH = 150.0


def setup_logging(log_level):
    """إعداد مستوى التسجيل بناءً على مدخلات المستخدم"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"خطأ: مستوى التسجيل '{log_level}' غير صالح.")
        sys.exit(1)
    logging.basicConfig(level=numeric_level, format='%(levelname)s: %(message)s')


def get_input_file_path(cli_path=None):
    """طلب مسار الملف تفاعلياً والتحقق من وجوده"""
    path = cli_path

    while not path:
        try:
            path = input("يرجى إدخال مسار ملف الصوت (MP3, OGG, WAV, M4A, إلخ): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nتم إيقاف العملية بواسطة المستخدم.")
            sys.exit(0)

    path = path.strip('"\'')

    while not os.path.isfile(path):
        print(f"خطأ: الملف '{path}' غير موجود أو المسار غير صحيح.")
        try:
            path = input("يرجى إدخال مسار صحيح للملف: ").strip().strip('"\'')
        except (KeyboardInterrupt, EOFError):
            print("\nتم إيقاف العملية بواسطة المستخدم.")
            sys.exit(0)

    return path


def estimate_pitch_accurate(y, sr):
    """
    حساب التردد الأساسي (F0) بدقة عالية مع إزالة الصمت والضوضاء
    """
    try:
        # 1. قص لحظات الصمت من التسجيل لضمان تحليل الصوت البشري فقط
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)

        # 2. استخدام pYIN لحساب ترددات الصوت البشري الممتدة بين 65Hz و 300Hz
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_trimmed,
            fmin=65,    # أدنى تردد للصوت البشري
            fmax=300,   # أقصى تردد لمعظم أشكال الحديث الطبيعي
            sr=sr
        )

        # 3. تصفية الإطارات واستخراج الترددات الحقيقية ذات الثقة العالية فقط (> 60%)
        if f0 is not None:
            valid_f0 = f0[voiced_flag & (voiced_probs > 0.6)]
            if len(valid_f0) > 0:
                # استخدام الوسيط الإحصائي (Median) لتجاهل الترددات الشاذة
                return float(np.median(valid_f0))
    except Exception as e:
        logging.debug(f"فشل حساب التردد الدقيق: {e}")

    return None


def process_audio(input_path, output_dir, steps_list):
    """معالجة الصوت وتعديل الطبقة مع اختيار أقرب نتيجة بدقة"""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"تم إنشاء مجلد المخرجات: {output_dir}")
        except PermissionError:
            print(f"خطأ: لا توجد صلاحيات لإنشاء المجلد '{output_dir}'.")
            return

    try:
        print(f"\nجاري تحميل الملف الصوتي: {input_path}")
        y, sr = librosa.load(input_path, sr=None)

        print(f"جاري معالجة {len(steps_list)} احتمالات مع تثبيت السرعة وتحليل النبرة...\n")

        best_option = None
        min_pitch_diff = float('inf')
        best_pitch_val = 0.0

        for step in steps_list:
            # تغيير طبقة الصوت بدون تغيير السرعة
            y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=step)

            sign = "+" if step > 0 else ""
            file_name = f"option_{sign}{step}_steps.wav"
            output_path = os.path.join(output_dir, file_name)

            # حفظ الملف
            sf.write(output_path, y_shifted, sr)

            # تحليل النبرة الناتجة بدقة
            pitch_f0 = estimate_pitch_accurate(y_shifted, sr)

            if pitch_f0 is not None:
                diff = abs(pitch_f0 - TARGET_HUMAN_PITCH)
                print(f"✓ تم حفظ: {file_name:<22} | التردد المقدر: {pitch_f0:.1f} Hz")

                if diff < min_pitch_diff:
                    min_pitch_diff = diff
                    best_option = file_name
                    best_pitch_val = pitch_f0
            else:
                print(f"✓ تم حفظ: {file_name:<22} | (تعذر قياس التردد بدقة)")

        print("\n" + "=" * 60)
        if best_option:
            print(f"🎯 التحديد التلقائي لأقرب نتيجة طبيعية:")
            print(f"   الملف المقترح: {best_option}")
            print(f"   التردد المحسوب: {best_pitch_val:.1f} Hz (ضمن نطاق الصوت البشري الطبيعي)")
        else:
            print("💡 لم يتم استنتاج الخيار الأفضل تلقائياً. يرجى الاستماع للملفات مباشرة.")
        print(f"📁 مجلد النتائج: '{output_dir}'")
        print("=" * 60)

    except FileNotFoundError:
        print(f"خطأ: الملف '{input_path}' غير موجود.")
    except PermissionError:
        print("خطأ: لا توجد صلاحيات لقراءة الملف أو الكتابة في المسار المحدد.")
    except Exception as e:
        print("حدث خطأ أثناء معالجة الملف الصوتي.")
        logging.debug(f"تفاصيل الخطأ: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="أداة تغيير طبقة الصوت وتحديد الخيار الأفضل بدقة - Librosa CLI",
        add_help=False
    )

    parser.add_argument('--help', action='help', help="إظهار رسالة المساعدة هذه والخروج")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="إظهار إصدار الأداة")
    parser.add_argument('--config', type=str, help="مسار ملف الإعدادات (JSON)")
    parser.add_argument('--log-level', type=str, default='INFO', help="مستوى التسجيل")

    parser.add_argument('-i', '--input', type=str, help="مسار ملف الصوت المدخل")
    parser.add_argument('-o', '--output', type=str, default="audio_options", help="مجلد المخرجات")

    args = parser.parse_args()
    setup_logging(args.log_level)

    steps_list = [-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8]
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                steps_list = config_data.get('steps', steps_list)
        except (FileNotFoundError, json.JSONDecodeError):
            print("تنبيه: تعذر قراءة ملف الإعدادات. سيتم استخدام القيم الافتراضية.")

    input_path = get_input_file_path(args.input)
    process_audio(input_path, args.output, steps_list)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nتم إيقاف العملية من قبل المستخدم.")