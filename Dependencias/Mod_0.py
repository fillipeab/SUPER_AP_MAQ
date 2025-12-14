# @title Módulo 0: Coordenador Principal - Arquitetura Modular com PersonDatabase
import cv2
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from .Mod_1 import *
from .Mod_2 import *

class Person:
    """
    Classe Person - Representa uma pessoa no sistema
    - ID único consistente
    - Histórico de IDs anteriores
    - Features modulares de diferentes algoritmos
    """

    def __init__(self, person_id: int, initial_track_id: int):
        self.id = person_id  # ID único e consistente no sistema
        self.previous_ids = [initial_track_id]  # Histórico de IDs de tracking
        self.features = {}  # Dicionário de features: {algoritmo: dados}
        self.stats = {
            'first_seen': None,
            'last_seen': None,
            'appearance_count': 0,
            'algorithms_used': set()
        }

    def add_previous_id(self, track_id: int):
        """Adiciona um ID anterior ao histórico"""
        if track_id not in self.previous_ids:
            self.previous_ids.append(track_id)

    def update_features(self, algorithm: str, features: Any):
        """Atualiza features de um algoritmo específico"""
        self.features[algorithm] = features
        self.stats['algorithms_used'].add(algorithm)

    def get_features(self, algorithm: str) -> Optional[Any]:
        """Obtém features de um algoritmo específico"""
        return self.features.get(algorithm)

    def has_features(self, algorithm: str) -> bool:
        """Verifica se tem features de um algoritmo"""
        return algorithm in self.features

    def update_appearance(self, frame_number: int):
        """Atualiza estatísticas de aparição"""
        if self.stats['first_seen'] is None:
            self.stats['first_seen'] = frame_number
        self.stats['last_seen'] = frame_number
        self.stats['appearance_count'] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para serialização"""
        return {
            'id': self.id,
            'previous_ids': self.previous_ids,
            'features_summary': {algo: f"features_{type(data).__name__}"
                               for algo, data in self.features.items()},
            'stats': self.stats
        }

    def __str__(self):
        return f"Person(ID: {self.id}, Previous: {self.previous_ids}, Algorithms: {list(self.features.keys())})"


class PersonDatabase:
    """
    Database de pessoas - Gerencia todas as pessoas no sistema
    """

    def __init__(self):
        self.people = {}  # {person_id: Person}
        self.track_id_to_person = {}  # Mapeamento: {track_id: person_id}
        self.next_person_id = 1

    def create_person(self, track_id: int) -> Person:
        """Cria uma nova pessoa"""
        person_id = self.next_person_id
        self.next_person_id += 1

        person = Person(person_id, track_id)
        self.people[person_id] = person
        self.track_id_to_person[track_id] = person_id

        return person

    def get_person_by_track_id(self, track_id: int) -> Optional[Person]:
        """Obtém pessoa por track_id atual"""
        person_id = self.track_id_to_person.get(track_id)
        return self.people.get(person_id) if person_id else None

    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """Obtém pessoa por ID do sistema"""
        return self.people.get(person_id)

    def update_track_id_mapping(self, old_track_id: int, new_track_id: int, person_id: int):
        """Atualiza mapeamento de track_id para person_id"""
        # Remover mapeamento antigo
        if old_track_id in self.track_id_to_person:
            del self.track_id_to_person[old_track_id]

        # Adicionar novo mapeamento
        self.track_id_to_person[new_track_id] = person_id

        # Atualizar histórico da pessoa
        person = self.get_person_by_id(person_id)
        if person:
            person.add_previous_id(new_track_id)

    def find_person_by_features(self, algorithm: str, features: Any,
                               similarity_func: callable, threshold: float) -> Optional[Person]:
        """Encontra pessoa baseado em features"""
        best_person = None
        best_similarity = 0.0

        for person in self.people.values():
            if person.has_features(algorithm):
                stored_features = person.get_features(algorithm)
                similarity = similarity_func(features, stored_features)

                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_person = person

        return best_person

    def get_all_people(self) -> List[Person]:
        """Retorna todas as pessoas"""
        return list(self.people.values())

    def get_people_with_features(self, algorithm: str) -> List[Person]:
        """Retorna pessoas que têm features de um algoritmo específico"""
        return [person for person in self.people.values()
                if person.has_features(algorithm)]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do database"""
        total_people = len(self.people)
        people_with_features = {}

        for person in self.people.values():
            for algo in person.features.keys():
                people_with_features[algo] = people_with_features.get(algo, 0) + 1

        return {
            'total_people': total_people,
            'total_track_mappings': len(self.track_id_to_person),
            'people_with_features': people_with_features,
            'next_person_id': self.next_person_id
        }

    def reset(self):
        """Reseta o database"""
        self.people = {}
        self.track_id_to_person = {}
        self.next_person_id = 1


class VideoProcessor:
    """
    Módulo 0: Coordenador Principal - Arquitetura Modular
    - Gerencia PersonDatabase central
    - Coordena módulos de processamento
    - Interface unificada para algoritmos
    """

    def __init__(self,
                 # ============================================================================
                 # 🔧 CONFIGURAÇÕES GERAIS DO SISTEMA
                 # ============================================================================
                 video_source=0,
                 output_base_dir="/content/results",
                 frame_skip=2,
                 max_processed_frames=50,
                 save_option=1,
                 show_preview=True,

                 # ============================================================================
                 # 🎯 CONFIGURAÇÕES DO MÓDULO 1 - YOLO TRACKER
                 # ============================================================================
                 model_size='n',
                 conf_threshold=0.3,
                 iou_threshold=0.5,
                 classes=[0],
                 persist=True,
                 tracker="bytetrack.yaml",

                 # ============================================================================
                 # 🎭 CONFIGURAÇÕES DO MÓDULO 2A - ReID SYSTEM
                 # ============================================================================
                 use_reid=True,
                 reid_config: Dict[str, Any] = None):
        """
        Inicializa o sistema com arquitetura modular
        """

        # ============================================================================
        # 🗂️ CONFIGURAÇÕES COMPLETAS
        # ============================================================================
        self.config = {
            # Configurações gerais
            'video_source': video_source,
            'output_base_dir': output_base_dir,
            'frame_skip': frame_skip,
            'max_processed_frames': max_processed_frames,
            'save_option': save_option,
            'show_preview': show_preview,

            # Configurações Módulo 1 - YOLO
            'model_size': model_size,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'classes': classes,
            'persist': persist,
            'tracker': tracker,

            # Configurações Módulo 2A - ReID
            'use_reid': use_reid,
            'reid_config': reid_config or {}
        }

        # ============================================================================
        # 🗃️ PERSON DATABASE (CENTRAL)
        # ============================================================================
        self.person_db = PersonDatabase()

        # ============================================================================
        # 🚀 INICIALIZAÇÃO DO SISTEMA
        # ============================================================================
        self._initialize_system()

    def _initialize_system(self):
        """Inicializa sistema modular completo"""
        try:
            print("🎯 Módulo 0 - Sistema Modular Inicializando...")

            # ✅ Diretório de resultados
            os.makedirs(self.config['output_base_dir'], exist_ok=True)

            # ✅ Módulo 1 - YOLO Tracker
            self.yolo_tracker = create_yolo_tracker(
                model_size=self.config['model_size'],
                conf_threshold=self.config['conf_threshold'],
                iou_threshold=self.config['iou_threshold'],
                classes=self.config['classes'],
                persist=self.config['persist'],
                tracker=self.config['tracker']
            )
            print("✅ Módulo 1 - YOLO Tracker inicializado")

            # ✅ Módulo 2A - ReID System (Opcional)
            self.reid_system = None
            if self.config['use_reid']:
                self.reid_system = create_reid_system(**self.config['reid_config'])
                print("✅ Módulo 2A - ReID System inicializado")
            else:
                print("⚡ Modo rápido: Apenas YOLO tracking")

            # ✅ Estado do sistema
            self.video_writers = {}
            self.is_webcam = self.config['video_source'] == 0
            self.current_session_dir = None
            self.processing_active = False
            self.frame_count = 0
            self.processed_count = 0
            self.start_time = None

            print("✅ Sistema modular completamente inicializado")
            self._print_initial_config()

        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            raise

    def _print_initial_config(self):
        """Exibe configuração inicial do sistema"""
        print(f"\n📋 CONFIGURAÇÃO DO SISTEMA MODULAR:")
        print(f"   🎥 Fonte: {self.config['video_source']}")
        print(f"   📁 Saída: {self.config['output_base_dir']}")
        print(f"   ⏩ Frame skip: {self.config['frame_skip']}")
        print(f"   📊 Frames máximos: {self.config['max_processed_frames']}")

        print(f"\n🎯 MÓDULO 1 - YOLO:")
        print(f"   • Modelo: yolov8{self.config['model_size']}")
        print(f"   • Confiança: {self.config['conf_threshold']}")

        print(f"\n🎭 MÓDULO 2A - ReID:")
        print(f"   • Ativo: {'SIM' if self.config['use_reid'] else 'NÃO'}")

        print(f"\n👥 PERSON DATABASE:")
        print(f"   • Pronto para gerenciar identidades")

    def process_video(self):
        """Processamento principal com arquitetura modular"""
        try:
            # ✅ Configurar vídeo
            cap = cv2.VideoCapture(self.config['video_source'])
            if not cap.isOpened():
                raise ValueError(f"❌ Não foi possível abrir: {self.config['video_source']}")

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else 0

            print(f"\n🎬 INICIANDO PROCESSAMENTO MODULAR:")
            print(f"   📏 Dimensões: {width}x{height}")
            print(f"   🎞️ FPS: {fps}")
            print(f"   👥 Person Database: Ativo")
            print(f"   🎭 ReID: {'LIGADO' if self.config['use_reid'] else 'DESLIGADO'}")

            # ✅ Gravação
            if self._should_save_video():
                session_dir = self._create_session_structure()
                self._setup_video_writers(session_dir, width, height, fps)

            # ✅ Loop principal
            return self._main_processing_loop(cap, fps, total_frames)

        except Exception as e:
            print(f"❌ Erro no processamento: {e}")
            return []

    def _main_processing_loop(self, cap, fps, total_frames):
        """Loop principal de processamento modular"""
        processed_frames = []
        self.frame_count = 0
        self.processed_count = 0
        self.processing_active = True
        self.start_time = time.time()

        try:
            while (self.processing_active and
                   self.processed_count < self.config['max_processed_frames']):

                ret, frame = cap.read()
                if not ret:
                    print("⏹️ Fim do vídeo alcançado")
                    break

                if self.frame_count % self.config['frame_skip'] == 0:
                    # ✅ Processar frame com arquitetura modular
                    processed_frame = self._process_frame_modular(frame, self.frame_count)

                    if processed_frame:
                        processed_frames.append(processed_frame)
                        self.processed_count += 1

                        # ✅ Salvar e mostrar
                        if self._should_save_video():
                            self._save_video_frames(processed_frame)

                        if self.config['show_preview']:
                            self._show_preview(processed_frame)

                        if self.processed_count % 10 == 0:
                            self._show_progress()

                self.frame_count += 1

                # ✅ Verificar fim do vídeo (arquivos)
                if not self.is_webcam and self.frame_count >= total_frames:
                    print("⏹️ Todos os frames do vídeo processados")
                    break

                # ✅ Parar com 'q'
                if self.config['show_preview'] and cv2.waitKey(1) & 0xFF == ord('q'):
                    self.processing_active = False
                    print("⏹️ Interrompido pelo usuário (tecla 'q')")

        except KeyboardInterrupt:
            print("\n⏹️ Interrompido pelo usuário (Ctrl+C)")
        except Exception as e:
            print(f"❌ Erro no loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            if self.config['show_preview']:
                cv2.destroyAllWindows()
            self._cleanup_video_writers()
            self._print_final_stats()

        return processed_frames

    def _process_frame_modular(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Processa frame com arquitetura modular
        - YOLO faz detecção inicial
        - Módulos processam através do PersonDatabase
        - PersonDatabase mantém estado consistente
        """
        try:
            # ============================================================================
            # 1️⃣ MÓDULO 1 - YOLO TRACKING (DETECÇÃO INICIAL)
            # ============================================================================
            yolo_results = self.yolo_tracker.process_frame(frame)

            if not yolo_results['success'] or yolo_results['num_detections'] == 0:
                return self._create_frame_result(frame, frame_number, [], False, yolo_results)

            # Preparar detecções no formato padrão
            yolo_detections = self._prepare_yolo_detections(yolo_results)

            # ============================================================================
            # 2️⃣ MÓDULO 2A - ReID PROCESSING (SE ATIVADO)
            # ============================================================================
            processed_detections = yolo_detections.copy()
            reid_applied = False
            reid_stats = {}

            if self.reid_system is not None:
                # 🎭 PROCESSAMENTO ReID COM PERSON DATABASE
                reid_results = self.reid_system.process_with_person_database(
                    frame=frame,
                    yolo_detections=yolo_detections,
                    person_db=self.person_db,
                    frame_number=frame_number
                )

                processed_detections = reid_results['processed_detections']
                reid_applied = True
                reid_stats = reid_results.get('reid_stats', {})

                print(f"   🎭 ReID: {reid_stats.get('matches', 0)} matches, {reid_stats.get('new_people', 0)} novos")

            # ============================================================================
            # 3️⃣ ATUALIZAR ESTATÍSTICAS DAS PESSOAS
            # ============================================================================
            self._update_people_appearances(processed_detections, frame_number)

            # ============================================================================
            # 4️⃣ CRIAR FRAME ANOTADO
            # ============================================================================
            annotated_frame = self._draw_frame_with_people(frame, processed_detections)

            # ============================================================================
            # 5️⃣ RESULTADO FINAL
            # ============================================================================
            return self._create_frame_result(
                frame, frame_number, processed_detections, reid_applied, yolo_results,
                annotated_frame, reid_stats
            )

        except Exception as e:
            print(f"⚠️ Erro no processamento modular do frame {frame_number}: {e}")
            return self._create_error_result(frame, frame_number)

    def _prepare_yolo_detections(self, yolo_results: Dict) -> List[Dict]:
        """Prepara detecções do YOLO no formato padrão"""
        detections = []
        for det in yolo_results['detections']:
            detections.append({
                'bbox': det['bbox'],
                'track_id': det['track_id'],
                'confidence': det['confidence'],
                'class_name': det['class_name'],
                'source': 'yolo'
            })
        return detections

    def _update_people_appearances(self, detections: List[Dict], frame_number: int):
        """Atualiza estatísticas de aparição das pessoas"""
        for det in detections:
            if 'person_id' in det:
                person_id = det['person_id']
                person = self.person_db.get_person_by_id(person_id)
                if person:
                    person.update_appearance(frame_number)

    def _draw_frame_with_people(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Desenha frame com informações das pessoas"""
        result_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det.get('track_id', -1)

            # 🎨 COR BASEADA NO person_id (se disponível)
            if 'person_id' in det:
                person_id = det['person_id']
                color = self._get_color_for_person(person_id)
                label = f"Person {person_id}"

                # Info adicional para pessoas ReID
                if det.get('id_source') == 'reid_match':
                    label += " (ReID)"
                    text_color = (0, 255, 0)
                else:
                    label += " (YOLO)"
                    text_color = (255, 0, 0)
            else:
                # Apenas YOLO
                color = (255, 0, 0)
                label = f"Track {track_id}"
                text_color = (255, 0, 0)

            # Bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Label
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Track ID (pequeno)
            cv2.putText(result_frame, f"Track: {track_id}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 🏷️ Informações do sistema
        self._draw_system_info(result_frame)

        return result_frame

    def _draw_system_info(self, frame: np.ndarray):
        """Desenha informações do sistema modular"""
        db_stats = self.person_db.get_stats()

        info_lines = [
            f"Modular System - Frame: {self.frame_count}",
            f"People: {db_stats['total_people']}",
            f"ReID: {'ON' if self.config['use_reid'] else 'OFF'}",
            f"Processing: {self.processed_count}/{self.config['max_processed_frames']}"
        ]

        for i, line in enumerate(info_lines):
            y_position = 30 + i * 25
            cv2.putText(frame, line, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _get_color_for_person(self, person_id: int) -> Tuple[int, int, int]:
        """Gera cor consistente baseada no person_id"""
        colors = [
            (255, 0, 0),    # Vermelho
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Ciano
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Amarelo
            (128, 0, 0),    # Vermelho escuro
            (0, 128, 0),    # Verde escuro
            (0, 0, 128),    # Azul escuro
        ]
        return colors[person_id % len(colors)]

    def _create_frame_result(self, frame: np.ndarray, frame_number: int,
                           detections: List[Dict], reid_applied: bool,
                           yolo_results: Dict, annotated_frame: np.ndarray = None,
                           reid_stats: Dict = None) -> Dict[str, Any]:
        """Cria resultado completo do frame processado"""
        if annotated_frame is None:
            annotated_frame = frame.copy()

        return {
            'frame_number': frame_number,
            'original_frame': frame,
            'annotated_frame': annotated_frame,
            'detections': detections,
            'reid_applied': reid_applied,
            'reid_stats': reid_stats or {},
            'yolo_raw': yolo_results,
            'person_db_stats': self.person_db.get_stats(),
            'processing_info': {
                'frame_skip': self.config['frame_skip'],
                'total_frames_read': self.frame_count,
                'total_frames_processed': self.processed_count
            }
        }

    def _create_error_result(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """Cria resultado de erro"""
        return {
            'frame_number': frame_number,
            'original_frame': frame,
            'annotated_frame': frame.copy(),
            'detections': [],
            'reid_applied': False,
            'reid_stats': {'error': True},
            'yolo_raw': None,
            'person_db_stats': self.person_db.get_stats(),
            'processing_info': {'error': True}
        }

    def _show_progress(self):
        """Mostra progresso do sistema modular"""
        elapsed = time.time() - self.start_time
        fps = self.processed_count / elapsed if elapsed > 0 else 0

        db_stats = self.person_db.get_stats()
        yolo_stats = self.yolo_tracker.get_tracking_stats()

        progress = f"⏳ Frame {self.frame_count} | Processados: {self.processed_count}/{self.config['max_processed_frames']}"
        people_info = f"👥 People: {db_stats['total_people']}"
        yolo_info = f"🎯 YOLO: {yolo_stats['unique_track_ids']} tracks"
        fps_info = f"⚡ FPS: {fps:.1f}"

        reid_info = ""
        if self.reid_system:
            reid_stats = self.reid_system.get_stats()
            reid_info = f" | 🎭 ReID: {reid_stats.get('reid_matches', 0)} matches"

        print(f"{progress} | {people_info} | {yolo_info} | {fps_info}{reid_info}")

    # ============================================================================
    # 🎯 MÉTODOS AUXILIARES (Mantidos da versão anterior)
    # ============================================================================

    def _should_save_video(self) -> bool:
        save_option = self.config['save_option']
        if save_option == 0:
            return self.is_webcam
        elif save_option == 1:
            return True
        elif save_option == 2:
            return False
        return True

    def _create_session_structure(self):
        base_dir = self.config['output_base_dir']
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(base_dir, f"session_{timestamp}")
        os.makedirs(session_dir)

        if not self.is_webcam:
            video_name = os.path.splitext(os.path.basename(str(self.config['video_source'])))[0]
            video_subdir = os.path.join(session_dir, video_name)
            os.makedirs(video_subdir)
            session_dir = video_subdir

        # Salvar configuração completa
        config_file = os.path.join(session_dir, "system_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        self.current_session_dir = session_dir
        print(f"📂 Sessão criada: {session_dir}")
        return session_dir

    def _setup_video_writers(self, session_dir: str, width: int, height: int, fps: float):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writers = {
                'original': cv2.VideoWriter(
                    os.path.join(session_dir, "01_original.mp4"), fourcc, fps, (width, height)
                ),
                'processed': cv2.VideoWriter(
                    os.path.join(session_dir, "02_processed.mp4"), fourcc, fps, (width, height)
                )
            }
            print(f"💾 Gravando vídeos em: {session_dir}/")
        except Exception as e:
            print(f"❌ Erro configurando video writers: {e}")
            self.video_writers = {}

    def _save_video_frames(self, processed_frame: Dict):
        try:
            if 'original' in self.video_writers:
                self.video_writers['original'].write(processed_frame['original_frame'])
            if 'processed' in self.video_writers:
                self.video_writers['processed'].write(processed_frame['annotated_frame'])
        except Exception as e:
            print(f"⚠️ Erro salvando frame: {e}")

    def _show_preview(self, processed_frame: Dict):
        try:
            cv2.imshow('Sistema Modular - YOLO + ReID + Person Database',
                      processed_frame['annotated_frame'])
        except Exception as e:
            print(f"⚠️ Erro no preview: {e}")

    def _cleanup_video_writers(self):
        for name, writer in self.video_writers.items():
            if writer is not None:
                writer.release()
                print(f"💾 Vídeo {name} finalizado")
        self.video_writers = {}
        if self.current_session_dir:
            print(f"💾 Todos os vídeos salvos em: {self.current_session_dir}")

    def _print_final_stats(self):
        """Exibe estatísticas finais do sistema modular"""
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.processed_count / total_time if total_time > 0 else 0

        print(f"\n{'='*60}")
        print(f"✅ PROCESSAMENTO MODULAR CONCLUÍDO")
        print(f"{'='*60}")

        print(f"📊 ESTATÍSTICAS GERAIS:")
        print(f"   • Frames lidos: {self.frame_count}")
        print(f"   • Frames processados: {self.processed_count}")
        print(f"   • Tempo total: {total_time:.2f}s")
        print(f"   • FPS médio: {avg_fps:.2f}")

        # Estatísticas do YOLO
        yolo_stats = self.yolo_tracker.get_tracking_stats()
        print(f"\n🎯 MÓDULO 1 - YOLO TRACKING:")
        print(f"   • Detecções totais: {yolo_stats['total_detections']}")
        print(f"   • Tracks únicos: {yolo_stats['unique_track_ids']}")
        print(f"   • Taxa de detecção: {yolo_stats['detection_rate_percentage']:.1f}%")

        # Estatísticas do Person Database
        db_stats = self.person_db.get_stats()
        print(f"\n👥 PERSON DATABASE:")
        print(f"   • Pessoas únicas: {db_stats['total_people']}")
        print(f"   • Mapeamentos: {db_stats['total_track_mappings']}")
        print(f"   • Próximo ID: {db_stats['next_person_id']}")

        # Estatísticas do ReID (se aplicável)
        if self.reid_system:
            reid_stats = self.reid_system.get_stats()
            print(f"\n🎭 MÓDULO 2A - ReID SYSTEM:")
            print(f"   • Matches ReID: {reid_stats.get('reid_matches', 0)}")
            print(f"   • Novas pessoas: {reid_stats.get('new_identities', 0)}")
            print(f"   • Extrações bem-sucedidas: {reid_stats.get('successful_extractions', 0)}")

        if self.current_session_dir:
            print(f"\n💾 Resultados salvos em: {self.current_session_dir}")

    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema modular"""
        yolo_stats = self.yolo_tracker.get_tracking_stats() if hasattr(self, 'yolo_tracker') else {}
        reid_stats = self.reid_system.get_stats() if self.reid_system else {}
        db_stats = self.person_db.get_stats()

        # Informações das primeiras 3 pessoas para debug
        people_sample = []
        all_people = self.person_db.get_all_people()
        for person in all_people[:3]:  # Primeiras 3 pessoas
            people_sample.append(person.to_dict())

        status = {
            'processing_active': self.processing_active,
            'frame_count': self.frame_count,
            'processed_count': self.processed_count,
            'current_session': self.current_session_dir,

            # Módulos
            'yolo_active': hasattr(self, 'yolo_tracker'),
            'reid_active': self.reid_system is not None,
            'reid_enabled': self.config['use_reid'],

            # Estatísticas
            'yolo_stats': yolo_stats,
            'reid_stats': reid_stats,
            'person_database': {
                'stats': db_stats,
                'people_sample': people_sample,
                'total_people': len(all_people)
            },

            'system_config': self.config
        }

        return status

    def stop_processing(self):
        """Para o processamento"""
        self.processing_active = False
        print("🛑 Parando processamento...")

    def reset_system(self):
        """Reseta todo o sistema modular"""
        self.processing_active = False
        self.frame_count = 0
        self.processed_count = 0

        if hasattr(self, 'yolo_tracker'):
            self.yolo_tracker.reset_stats()

        if self.reid_system:
            self.reid_system.reset()

        self.person_db.reset()

        print("🔄 Sistema modular completamente resetado")


# ============================================================================
# 🚀 FUNÇÃO DE CRIAÇÃO
# ============================================================================
def create_video_processor(
    # Configurações Gerais
    video_source=0,
    output_base_dir="/content/results",
    frame_skip=2,
    max_processed_frames=50,
    save_option=1,
    show_preview=True,

    # Configurações Módulo 1 - YOLO
    model_size='n',
    conf_threshold=0.3,
    iou_threshold=0.5,
    classes=[0],
    persist=True,
    tracker="bytetrack.yaml",

    # Configurações Módulo 2A - ReID
    use_reid=True,
    reid_config: Dict[str, Any] = None
) -> VideoProcessor:
    """
    Cria instância do VideoProcessor com arquitetura modular
    """
    return VideoProcessor(
        # Configurações Gerais
        video_source=video_source,
        output_base_dir=output_base_dir,
        frame_skip=frame_skip,
        max_processed_frames=max_processed_frames,
        save_option=save_option,
        show_preview=show_preview,

        # Configurações Módulo 1
        model_size=model_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
        persist=persist,
        tracker=tracker,

        # Configurações Módulo 2A
        use_reid=use_reid,
        reid_config=reid_config or {}
    )


