# @title Módulo 1: YOLO Tracker - Detecção e Tracking
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Optional, Tuple

class YOLOTracker:
    """
    Módulo 1: Especializado em detecção e tracking YOLO
    - Gerencia o modelo YOLO
    - Aplica tracking com persistência
    - Processa frames individuais
    - Fornece estatísticas de tracking
    """

    def __init__(self,
                 model_size: str = 'n',
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.5,
                 classes: List[int] = None,
                 persist: bool = True,
                 tracker: str = "bytetrack.yaml"):

        # ============================================================================
        # 🗂️ CONFIGURAÇÕES
        # ============================================================================
        self.config = {
            'model_size': model_size,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'classes': classes or [0],  # Default: apenas pessoas
            'persist': persist,
            'tracker': tracker
        }

        # ============================================================================
        # 📊 ESTATÍSTICAS
        # ============================================================================
        self.stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'unique_track_ids': set(),
            'frames_with_detections': 0,
            'tracking_history': {}  # Histórico por ID
        }

        # ============================================================================
        # 🚀 INICIALIZAR MODELO YOLO
        # ============================================================================
        self._initialize_model()

        print("🎯 Módulo 1 - YOLO Tracker Inicializado")

    def _initialize_model(self):
        """
        Inicializa o modelo YOLO com as configurações especificadas
        """
        try:
            print("🔄 Carregando modelo YOLO...")

            # Carregar modelo baseado no tamanho
            model_name = f'yolov8{self.config["model_size"]}.pt'
            self.model = YOLO(model_name)

            # Verificar se GPU está disponível
            #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #print(f"   📦 Modelo: {model_name}")
            #print(f"   ⚡ Dispositivo: {self.device.upper()}")
            #print(f"   🎯 Classes: {self.config['classes']}")
            #print(f"   🔄 Persist: {self.config['persist']}")
            #print(f"   🏷️ Tracker: {self.config['tracker']}")

        except Exception as e:
            print(f"❌ Erro ao carregar modelo YOLO: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Processa um único frame com YOLO tracking

        Args:
            frame: Frame BGR do OpenCV

        Returns:
            Dict com resultados do processamento
        """
        try:
            # ============================================================================
            # 🎯 EXECUTAR YOLO TRACKING
            # ============================================================================
            results = self.model.track(
                frame,
                persist=self.config['persist'],
                tracker=self.config['tracker'],
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                classes=self.config['classes'],
                verbose=False  # Silencioso
            )

            # ============================================================================
            # 📊 PROCESSAR RESULTADOS
            # ============================================================================
            processed_data = self._process_results(results, frame)

            # ============================================================================
            # 📈 ATUALIZAR ESTATÍSTICAS
            # ============================================================================
            self._update_stats(processed_data)

            return processed_data

        except Exception as e:
            print(f"⚠️ Erro no processamento do frame: {e}")
            return self._create_empty_result(frame)

    def _process_results(self, results, original_frame: np.ndarray) -> Dict:
        """
        Processa os resultados do YOLO e extrai informações
        """
        if not results or len(results) == 0:
            return self._create_empty_result(original_frame)

        result = results[0]

        # ============================================================================
        # 🎯 EXTRAIR DETECÇÕES E TRACKING
        # ============================================================================
        detections = []
        track_ids = []

        if result.boxes is not None:
            # Converter para CPU e numpy
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []

            # IDs de tracking (se disponíveis)
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int).tolist()
            else:
                track_ids = [-1] * len(boxes)  # -1 indica sem tracking

            # Criar lista de detecções
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                track_id = track_ids[i] if i < len(track_ids) else -1

                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': self.model.names[int(cls_id)],
                    'track_id': int(track_id),
                    'center': self._calculate_center(box)
                }
                detections.append(detection)

        # ============================================================================
        # 🖼️ CRIAR FRAME ANOTADO
        # ============================================================================
        annotated_frame = result.plot() if hasattr(result, 'plot') else original_frame.copy()

        # ============================================================================
        # 📦 ESTRUTURAR RESULTADO
        # ============================================================================
        return {
            'success': True,
            'original_frame': original_frame,
            'annotated_frame': annotated_frame,
            'detections': detections,
            'track_ids': track_ids,
            'num_detections': len(detections),
            'has_tracking': len(track_ids) > 0 and any(tid != -1 for tid in track_ids),
            'yolo_raw_result': result
        }

    def _calculate_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        Calcula o centro de uma bounding box
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return float(center_x), float(center_y)

    def _create_empty_result(self, frame: np.ndarray) -> Dict:
        """
        Cria resultado vazio para frames sem detecções
        """
        return {
            'success': False,
            'original_frame': frame,
            'annotated_frame': frame.copy(),
            'detections': [],
            'track_ids': [],
            'num_detections': 0,
            'has_tracking': False,
            'yolo_raw_result': None
        }

    def _update_stats(self, processed_data: Dict):
        """
        Atualiza estatísticas de tracking
        """
        self.stats['total_frames_processed'] += 1

        if processed_data['success'] and processed_data['num_detections'] > 0:
            self.stats['total_detections'] += processed_data['num_detections']
            self.stats['frames_with_detections'] += 1

            # Atualizar IDs únicos
            for detection in processed_data['detections']:
                if detection['track_id'] != -1:
                    self.stats['unique_track_ids'].add(detection['track_id'])

                    # Atualizar histórico de tracking
                    self._update_tracking_history(detection)

    def _update_tracking_history(self, detection: Dict):
        """
        Atualiza o histórico de tracking para um ID específico
        """
        track_id = detection['track_id']

        if track_id not in self.stats['tracking_history']:
            self.stats['tracking_history'][track_id] = {
                'first_seen': self.stats['total_frames_processed'],
                'last_seen': self.stats['total_frames_processed'],
                'total_appearances': 0,
                'class_counts': {},
                'positions': []
            }

        history = self.stats['tracking_history'][track_id]
        history['last_seen'] = self.stats['total_frames_processed']
        history['total_appearances'] += 1

        # Contar classes
        class_name = detection['class_name']
        history['class_counts'][class_name] = history['class_counts'].get(class_name, 0) + 1

        # Registrar posição
        history['positions'].append({
            'frame': self.stats['total_frames_processed'],
            'center': detection['center'],
            'bbox': detection['bbox']
        })

    def get_tracking_stats(self) -> Dict:
        """
        Retorna estatísticas completas de tracking
        """
        detection_rate = (self.stats['frames_with_detections'] /
                         max(self.stats['total_frames_processed'], 1)) * 100

        avg_detections = (self.stats['total_detections'] /
                         max(self.stats['frames_with_detections'], 1))

        return {
            'total_frames_processed': self.stats['total_frames_processed'],
            'total_detections': self.stats['total_detections'],
            'unique_track_ids': len(self.stats['unique_track_ids']),
            'frames_with_detections': self.stats['frames_with_detections'],
            'detection_rate_percentage': detection_rate,
            'average_detections_per_frame': avg_detections,
            'active_tracks': len([tid for tid, hist in self.stats['tracking_history'].items()
                                if self.stats['total_frames_processed'] - hist['last_seen'] < 10])
        }

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Retorna informações específicas de um track ID
        """
        return self.stats['tracking_history'].get(track_id)

    def get_active_tracks(self) -> List[int]:
        """
        Retorna lista de tracks ativos (vistos nos últimos 10 frames)
        """
        current_frame = self.stats['total_frames_processed']
        active_tracks = []

        for track_id, history in self.stats['tracking_history'].items():
            if current_frame - history['last_seen'] < 10:  # Considerado ativo se visto nos últimos 10 frames
                active_tracks.append(track_id)

        return active_tracks

    def reset_stats(self):
        """
        Reseta todas as estatísticas
        """
        self.stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'unique_track_ids': set(),
            'frames_with_detections': 0,
            'tracking_history': {}
        }
        print("📊 Estatísticas do YOLO Tracker resetadas")

    def get_config(self) -> Dict:
        """
        Retorna configuração atual
        """
        return self.config.copy()

    def __del__(self):
        """
        Destrutor - limpeza de recursos
        """
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 🚀 FUNÇÃO DE CRIAÇÃO SIMPLIFICADA
# ============================================================================
def create_yolo_tracker(**kwargs):
    """
    Cria uma instância do YOLOTracker com configurações personalizadas

    Exemplo de uso:
    tracker = create_yolo_tracker(
        model_size='m',
        conf_threshold=0.5,
        persist=True,
        classes=[0, 2]  # Pessoas e carros
    )
    """
    return YOLOTracker(**kwargs)
