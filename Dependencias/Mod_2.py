#
@title Módulo 2A: ReID System - Adaptado para Arquitetura Modular
,import numpy as np6+03'

2,import cv2
import torch
import torchreid
from typing import Dict, List, Optional, Tuple, Any
import time
from PIL import Image

class ReIDSystem:
    """
    Módulo 2A: Sistema de Re-Identification
    - Adaptado para trabalhar com PersonDatabase
    - Interface modular com Módulo 0
    - Mantém compatibilidade com versão anterior
    """

    def __init__(self,
                 similarity_threshold: float = 0.7,
                 max_features_per_id: int = 5,
                 use_gpu: bool = True,
                 model_name: str = 'osnet_x1_0',
                 enable_temporal_smoothing: bool = True,
                 extraction_size: Tuple[int, int] = (256, 128)):
        """
        Inicializa com configuração flexível
        """

        # ============================================================================
        # 🔧 CONFIGURAÇÃO FLEXÍVEL
        # ============================================================================
        self.config = {
            'similarity_threshold': similarity_threshold,
            'max_features_per_id': max_features_per_id,
            'use_gpu': use_gpu,
            'model_name': model_name,
            'enable_temporal_smoothing': enable_temporal_smoothing,
            'extraction_size': extraction_size
        }

        # ============================================================================
        # 🗃️ ESTADO INTERNO
        # ============================================================================
        self.reid_model = None
        self.transform = None

        # Estatísticas
        self.stats = {
            'frames_processed': 0,
            'reid_matches': 0,
            'new_identities': 0,
            'id_corrections': 0,
            'total_processing_time': 0.0,
            'successful_extractions': 0,
            'extraction_errors': 0
        }

        # ============================================================================
        # 🚀 INICIALIZAÇÃO
        # ============================================================================
        self._initialize_reid_model()

        print(f"🎭 Módulo 2A - ReID System (Modular) Inicializado")
        print(f"   ⚙️  Similarity: {self.config['similarity_threshold']}")
        print(f"   📦 Model: {self.config['model_name']}")

    def _initialize_reid_model(self):
        """Inicializa o modelo ReID"""
        try:
            print("🔄 Carregando modelo ReID...")

            # Dispositivo
            self.device = 'cuda' if torch.cuda.is_available() and self.config['use_gpu'] else 'cpu'

            # Carregar modelo
            self.reid_model = torchreid.models.build_model(
                name=self.config['model_name'],
                num_classes=1,
                pretrained=True
            )
            self.reid_model.to(self.device)
            self.reid_model.eval()

            # Inicializar transformações
            self._initialize_transforms()

            print(f"   ✅ Modelo carregado em {self.device}")

        except Exception as e:
            print(f"❌ Erro ao carregar modelo ReID: {e}")
            self.reid_model = None
            self.transform = None

    def _initialize_transforms(self):
        """Inicializa transformações"""
        try:
            # Usar transforms do torchreid
            transforms_list = torchreid.data.transforms.build_transforms(
                height=self.config['extraction_size'][1],
                width=self.config['extraction_size'][0],
                random_erase=False,
                color_jitter=False,
                color_aug=False
            )
            self.transform = transforms_list[0]
            print("   ✅ Transformações carregadas")

        except Exception as e:
            print(f"❌ Erro nas transformações: {e}")
            self.transform = None

    def process_with_person_database(self,
                                   frame: np.ndarray,
                                   yolo_detections: List[Dict],
                                   person_db: Any,  # PersonDatabase do Módulo 0
                                   frame_number: int) -> Dict[str, Any]:
        """
        🆕 NOVA INTERFACE: Processa usando PersonDatabase compartilhado

        Args:
            frame: Frame BGR do OpenCV
            yolo_detections: Lista de detecções do YOLO
            person_db: Instância de PersonDatabase do Módulo 0
            frame_number: Número do frame para tracking

        Returns:
            Dict com resultados processados
        """
        start_time = time.time()

        try:
            if not yolo_detections or not self._is_reid_available():
                return {
                    'processed_detections': yolo_detections,
                    'annotated_frame': self._draw_yolo_only(frame, yolo_detections),
                    'reid_stats': {'reid_applied': False}
                }

            # ============================================================================
            # 1️⃣ EXTRAIR FEATURES DAS DETECÇÕES
            # ============================================================================
            features_data = self._extract_features_from_detections(frame, yolo_detections)

            if not features_data:
                return {
                    'processed_detections': yolo_detections,
                    'annotated_frame': self._draw_yolo_only(frame, yolo_detections),
                    'reid_stats': {'reid_applied': False, 'error': 'no_features'}
                }

            # ============================================================================
            # 2️⃣ PROCESSAR COM PERSON DATABASE
            # ============================================================================
            processed_detections = self._process_with_database(
                features_data, yolo_detections, person_db, frame_number
            )

            # ============================================================================
            # 3️⃣ DESENHAR RESULTADOS
            # ============================================================================
            annotated_frame = self._draw_reid_results_with_people(frame, processed_detections)

            # ============================================================================
            # 📊 ESTATÍSTICAS
            # ============================================================================
            processing_time = time.time() - start_time
            stats = self._get_processing_stats(processing_time, len(features_data))

            return {
                'processed_detections': processed_detections,
                'annotated_frame': annotated_frame,
                'reid_stats': stats
            }

        except Exception as e:
            print(f"❌ Erro no ReID com database: {e}")
            return {
                'processed_detections': yolo_detections,
                'annotated_frame': self._draw_yolo_only(frame, yolo_detections),
                'reid_stats': {'reid_applied': False, 'error': str(e)}
            }

    def _process_with_database(self,
                             features_data: List[Dict],
                             yolo_detections: List[Dict],
                             person_db: Any,
                             frame_number: int) -> List[Dict]:
        """Processa detecções usando PersonDatabase"""
        processed_detections = yolo_detections.copy()

        for feature_item in features_data:
            idx = feature_item['detection_index']
            features = feature_item['features']
            track_id = feature_item['original_track_id']

            # 🆕 BUSCAR PESSOA EXISTENTE POR FEATURES
            existing_person = person_db.find_person_by_features(
                algorithm='reid',
                features=features,
                similarity_func=self._cosine_similarity,
                threshold=self.config['similarity_threshold']
            )

            if existing_person:
                # ✅ PESSOA EXISTENTE ENCONTRADA
                processed_detections[idx]['person_id'] = existing_person.id
                processed_detections[idx]['id_source'] = 'reid_match'

                # Atualizar features da pessoa
                existing_person.update_features('reid', features)

                # Atualizar mapeamento track_id -> person_id
                person_db.update_track_id_mapping(
                    old_track_id=track_id,
                    new_track_id=track_id,
                    person_id=existing_person.id
                )

                self.stats['reid_matches'] += 1
                print(f"   🔄 Pessoa {existing_person.id} reconhecida (Track {track_id})")

            else:
                # 🆕 NOVA PESSOA
                new_person = person_db.create_person(track_id)
                new_person.update_features('reid', features)

                processed_detections[idx]['person_id'] = new_person.id
                processed_detections[idx]['id_source'] = 'new_person'

                self.stats['new_identities'] += 1
                print(f"   👤 Nova pessoa {new_person.id} criada (Track {track_id})")

        return processed_detections

    def _extract_features_from_detections(self,
                                        frame: np.ndarray,
                                        detections: List[Dict]) -> List[Dict]:
        """Extrai features ReID para cada detecção"""
        features_list = []

        for i, det in enumerate(detections):
            try:
                # Extrair ROI da pessoa
                x1, y1, x2, y2 = map(int, det['bbox'])

                # Garantir coordenadas válidas
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size == 0:
                    continue

                # Converter BGR para RGB
                person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(person_roi_rgb)

                # Aplicar transformações
                processed_img = self.transform(pil_image)

                # Garantir que é tensor e adicionar batch dimension
                if not isinstance(processed_img, torch.Tensor):
                    processed_img = torch.tensor(processed_img)

                processed_img = processed_img.unsqueeze(0).to(self.device)

                # Extrair features
                with torch.no_grad():
                    features = self.reid_model(processed_img)
                    features = features.cpu().numpy().flatten()

                # Normalizar features
                features = features / (np.linalg.norm(features) + 1e-8)

                features_list.append({
                    'features': features,
                    'detection_index': i,
                    'bbox': [x1, y1, x2, y2],
                    'original_track_id': det.get('track_id', -1),
                    'confidence': det.get('confidence', 0.0)
                })

                self.stats['successful_extractions'] += 1

            except Exception as e:
                print(f"⚠️  Erro extraindo features detecção {i}: {e}")
                self.stats['extraction_errors'] += 1
                continue

        return features_list

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similaridade cosseno"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def _draw_reid_results_with_people(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Desenha resultados com person_id"""
        result_frame = frame.copy()

        for det in detections:
            if 'person_id' not in det:
                continue

            x1, y1, x2, y2 = map(int, det['bbox'])
            person_id = det['person_id']
            track_id = det.get('track_id', -1)
            id_source = det.get('id_source', 'unknown')

            # Cor baseada no person_id
            color = self._get_color_for_person(person_id)

            # Bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Texto do ID
            if id_source == 'reid_match':
                label = f"Person {person_id} (ReID)"
                text_color = (0, 255, 0)  # Verde para ReID
            else:
                label = f"Person {person_id} (New)"
                text_color = (0, 255, 255)  # Amarelo para novo

            # Fundo para texto
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)

            # Texto
            cv2.putText(result_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Track ID original
            if track_id != -1:
                yolo_label = f"Track: {track_id}"
                cv2.putText(result_frame, yolo_label, (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Informações do sistema
        self._draw_system_info(result_frame)

        return result_frame

    def _draw_yolo_only(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Desenha apenas detecções YOLO (fallback)"""
        result_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det.get('track_id', -1)

            # Bounding box azul para YOLO puro
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if track_id != -1:
                label = f"Track {track_id} (YOLO)"
                cv2.putText(result_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(result_frame, "ReID: OFF", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return result_frame

    def _draw_system_info(self, frame: np.ndarray):
        """Adiciona informações do sistema"""
        info_lines = [
            f"ReID System - Matches: {self.stats['reid_matches']}",
            f"New People: {self.stats['new_identities']}",
            f"Extractions: {self.stats['successful_extractions']}",
            f"Frames: {self.stats['frames_processed']}"
        ]

        for i, line in enumerate(info_lines):
            y_position = 30 + i * 20
            cv2.putText(frame, line, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _get_color_for_person(self, person_id: int) -> Tuple[int, int, int]:
        """Gera cor consistente baseada no person_id"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[person_id % len(colors)]

    def _get_processing_stats(self, processing_time: float, num_features: int) -> Dict[str, Any]:
        """Retorna estatísticas do processamento"""
        self.stats['frames_processed'] += 1
        self.stats['total_processing_time'] += processing_time

        avg_time = (self.stats['total_processing_time'] /
                   max(self.stats['frames_processed'], 1))

        return {
            'reid_applied': True,
            'processing_time': processing_time,
            'avg_processing_time': avg_time,
            'matches': self.stats['reid_matches'],
            'new_people': self.stats['new_identities'],
            'successful_extractions': self.stats['successful_extractions'],
            'extraction_errors': self.stats['extraction_errors']
        }

    def _is_reid_available(self) -> bool:
        """Verifica se o ReID está disponível para uso"""
        return self.reid_model is not None and self.transform is not None

    # ============================================================================
    # 🎯 MÉTODOS DE COMPATIBILIDADE (para uso com versão anterior se necessário)
    # ============================================================================

    def process_yolo_detections(self,
                               frame: np.ndarray,
                               yolo_detections: List[Dict],
                               frame_number: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        🎯 MÉTODO DE COMPATIBILIDADE: Interface antiga para manter compatibilidade
        """
        print("⚠️  Usando interface legada - migre para process_with_person_database()")

        # Simular comportamento antigo (sem PersonDatabase)
        result_frame = self._draw_yolo_only(frame, yolo_detections)
        return result_frame, yolo_detections

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema"""
        avg_time = (self.stats['total_processing_time'] /
                   max(self.stats['frames_processed'], 1))

        return {
            'frames_processed': self.stats['frames_processed'],
            'reid_matches': self.stats['reid_matches'],
            'new_identities': self.stats['new_identities'],
            'id_corrections': self.stats['id_corrections'],
            'avg_processing_time': avg_time,
            'successful_extractions': self.stats['successful_extractions'],
            'extraction_errors': self.stats['extraction_errors']
        }

    def get_config(self) -> Dict[str, Any]:
        """Retorna configuração atual"""
        return self.config.copy()

    def cleanup(self):
        """Limpeza de recursos"""
        if hasattr(self, 'reid_model'):
            del self.reid_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self):
        """Reseta o sistema completamente"""
        self.stats = {
            'frames_processed': 0,
            'reid_matches': 0,
            'new_identities': 0,
            'id_corrections': 0,
            'total_processing_time': 0.0,
            'successful_extractions': 0,
            'extraction_errors': 0
        }
        print("🔄 ReID System resetado")


# ============================================================================
# 🚀 FUNÇÃO DE CRIAÇÃO
# ============================================================================
def create_reid_system(
    similarity_threshold: float = 0.7,
    max_features_per_id: int = 5,
    use_gpu: bool = True,
    model_name: str = 'osnet_x1_0',
    enable_temporal_smoothing: bool = True,
    extraction_size: Tuple[int, int] = (256, 128)
) -> ReIDSystem:
    """
    Cria uma instância configurável do ReIDSystem

    Args:
        similarity_threshold: Limite para match (0.0-1.0)
        max_features_per_id: Máximo de features por ID
        use_gpu: Usar GPU se disponível
        model_name: Nome do modelo ReID
        enable_temporal_smoothing: Suavização temporal
        extraction_size: Tamanho para extração (width, height)

    Returns:
        Instância configurada do ReIDSystem
    """
    return ReIDSystem(
        similarity_threshold=similarity_threshold,
        max_features_per_id=max_features_per_id,
        use_gpu=use_gpu,
        model_name=model_name,
        enable_temporal_smoothing=enable_temporal_smoothing,
        extraction_size=extraction_size
    )
