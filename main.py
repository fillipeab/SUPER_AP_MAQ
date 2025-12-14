import Dependencias as dpn
dpn.set_env()


### Função auxiliar para definir a pasta de saida
import os
def get_output_dir(folder_name="arquivos/resultados"):
    """Retorna o caminho para a pasta de resultados no mesmo diretório do script"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print (output_dir)
    return output_dir


def main():
    get_output_dir()
    # Exemplo com arquitetura modular

    # Configuração ReID
    reid_config = {
        'similarity_threshold': 0.7,
        'max_features_per_id': 3,
        'model_name': 'osnet_x0_25'
    }

    # Criar processor
    processor = dpn.create_video_processor(
        video_source=0,
        output_base_dir=get_output_dir(),
        model_size='n',
        conf_threshold=0.4,
        use_reid=True,
        reid_config=reid_config,
        max_processed_frames=240,
        show_preview=True,
        frame_skip=1
    )

    # Iniciar processamento
    print("🎬 Iniciando processamento modular...")
    results = processor.process_video()

    # Status final
    status = processor.get_system_status()
    print(f"\n📈 Status final do sistema modular:")
    print(f"   • Frames processados: {status['processed_count']}")
    print(f"   • Pessoas no database: {status['person_database']['total_people']}")
    print(f"   • ReID ativo: {status['reid_active']}")

    return 0

main()
