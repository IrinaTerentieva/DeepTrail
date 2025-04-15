import os
import hydra
from omegaconf import DictConfig
from keras_unet.models import custom_unet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.data_generator import DataGenerator
from src.metrics import iou, iou_thresholded
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(patch_dir):
    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patch_dir, f) for f in all_files if 'img' in f]
    train_labels = [os.path.join(patch_dir, f.replace('img', 'lab')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('img', 'lab')))]
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    return train_images, val_images, train_labels, val_labels

def train_model(cfg: DictConfig):
    input_shape = (cfg.training.size, cfg.training.size, 1)
    print(f"Input shape: {input_shape}")

    model = custom_unet(
        input_shape=input_shape,
        filters=cfg.training.filters,
        use_batch_norm=True,
        dropout=0.3,
        num_classes=1,
        output_activation='sigmoid'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss='binary_crossentropy',
        metrics=[iou, iou_thresholded]
    )

    earlystopper = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)

    checkpoint_filepath = os.path.join(cfg.training.checkpoint_dir, f"trails_tracks_model_epoch_{{epoch:02d}}_valloss_{{val_loss:.2f}}.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    train_gen = DataGenerator(
        image_list=cfg.train_images,
        mask_list=cfg.train_labels,
        batch_size=cfg.training.batch_size,
        image_size=(cfg.training.size, cfg.training.size),
        shuffle=True,
        min_area=cfg.training.min_area,
        buffer_size=cfg.training.buffer_size,
        threshold=cfg.training.threshold
    )

    val_gen = DataGenerator(
        image_list=cfg.val_images,
        mask_list=cfg.val_labels,
        batch_size=cfg.training.batch_size,
        image_size=(cfg.training.size, cfg.training.size),
        shuffle=False,
        min_area=cfg.training.min_area,
        buffer_size=cfg.training.buffer_size,
        threshold=cfg.training.threshold
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.training.epochs,
        steps_per_epoch=len(train_gen) // cfg.training.batch_size,
        validation_steps=len(val_gen) // cfg.training.batch_size,
        callbacks=[earlystopper, reduce_lr, checkpoint_callback]
    )

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    train_images, val_images, train_labels, val_labels = load_data(cfg.training.patch_dir)
    cfg.train_images = train_images
    cfg.val_images = val_images
    cfg.train_labels = train_labels
    cfg.val_labels = val_labels
    train_model(cfg)

if __name__ == "__main__":
    main()
