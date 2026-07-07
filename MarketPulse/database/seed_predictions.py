import datetime
import random
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import Stock, MarketData, Prediction, ModelRun

def seed_predictions():
    db = SessionLocal()
    try:
        # Create Model Runs
        print("Seeding model runs...")
        runs = [
            ModelRun(
                model_name="vit_b_16",
                config_name="vit_head_lr_3e-4",
                test_accuracy=0.824,
                test_macro_f1=0.814,
                test_weighted_f1=0.825,
                test_loss=0.412,
                status="SUCCESS",
                checkpoint_path="data/models/vit_b_16_vit_head_lr_3e-4.pth"
            ),
            ModelRun(
                model_name="resnet18",
                config_name="resnet_finetune_lr_1e-4",
                test_accuracy=0.801,
                test_macro_f1=0.785,
                test_weighted_f1=0.796,
                test_loss=0.456,
                status="SUCCESS",
                checkpoint_path="data/models/resnet18_resnet_finetune_lr_1e-4.pth"
            ),
            ModelRun(
                model_name="custom_cnn",
                config_name="cnn_lr_1e-3",
                test_accuracy=0.735,
                test_macro_f1=0.710,
                test_weighted_f1=0.722,
                test_loss=0.621,
                status="SUCCESS",
                checkpoint_path="data/models/custom_cnn_cnn_lr_1e-3.pth"
            )
        ]
        
        for r in runs:
            existing = db.query(ModelRun).filter(ModelRun.model_name == r.model_name, ModelRun.config_name == r.config_name).first()
            if not existing:
                db.add(r)
        db.commit()

        # Seed predictions based on market data timestamps
        print("Seeding predictions table...")
        stocks = db.query(Stock).all()
        model_run = db.query(ModelRun).filter(ModelRun.model_name == "vit_b_16").first()
        
        predictions_seeded = 0
        for stock in stocks:
            md_list = db.query(MarketData).filter(MarketData.stock_id == stock.id).order_by(MarketData.timestamp.desc()).limit(50).all()
            if not md_list:
                continue
                
            for idx, md in enumerate(md_list):
                # Check duplicate
                existing_pred = db.query(Prediction).filter(
                    Prediction.stock_id == stock.id,
                    Prediction.prediction_timestamp == md.timestamp
                ).first()
                
                if not existing_pred:
                    # Deterministic mock predictions
                    labels = ["up", "down", "neutral"]
                    pred_lbl = labels[(idx + stock.id) % 3]
                    
                    # 80% accuracy for realistic looking records
                    is_correct = (random.random() < 0.8)
                    true_lbl = pred_lbl if is_correct else labels[(idx + stock.id + 1) % 3]
                    
                    pred = Prediction(
                        stock_id=stock.id,
                        model_run_id=model_run.id if model_run else None,
                        prediction_timestamp=md.timestamp,
                        label_date=md.timestamp.date(),
                        current_close=md.close,
                        next_close=md.close * (1.0 + (0.012 if true_lbl == "up" else -0.012 if true_lbl == "down" else 0.0)),
                        next_day_return=(0.012 if true_lbl == "up" else -0.012 if true_lbl == "down" else 0.0),
                        predicted_label=pred_lbl,
                        true_label=true_lbl,
                        confidence=0.72 + (idx % 5) * 0.05,
                        correct=is_correct
                    )
                    db.add(pred)
                    predictions_seeded += 1
                    
        db.commit()
        print(f"Seed complete. Injected {predictions_seeded} prediction rows.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding predictions: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_predictions()
