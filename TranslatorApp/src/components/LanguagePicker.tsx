import React, { useState } from "react";
import { View, StyleSheet, Text, TouchableOpacity, Modal, FlatList } from "react-native";

type LanguagePickerProps = {
	selected: string;
	onSelect: (lang: string) => void;
};

const languages = [
	"English (UK)",
	"French",
	"German",
	"Spanish",
	"Hindi",
	"Telugu",
	"Tamil",
];



export function LanguagePicker({ selected, onSelect }: LanguagePickerProps) {
	const [modalVisible, setModalVisible] = useState(false);

	return (
		<View style={styles.container}>
			<Text style={styles.label}>Select Language:</Text>
			<TouchableOpacity
				style={styles.picker}
				onPress={() => setModalVisible(true)}
			>
				<Text>{selected}</Text>
			</TouchableOpacity>
			<Modal
				visible={modalVisible}
				transparent
				animationType="slide"
				onRequestClose={() => setModalVisible(false)}
			>
				<View style={styles.modalOverlay}>
					<View style={styles.modalContent}>
						<FlatList
							data={languages}
							keyExtractor={(item) => item}
							renderItem={({ item }) => (
								<TouchableOpacity
									style={styles.option}
									onPress={() => {
										onSelect(item);
										setModalVisible(false);
									}}
								>
									<Text style={selected === item ? styles.selected : undefined}>{item}</Text>
								</TouchableOpacity>
							)}
						/>
						<TouchableOpacity onPress={() => setModalVisible(false)} style={styles.closeBtn}>
							<Text style={{ color: '#fff' }}>Close</Text>
						</TouchableOpacity>
					</View>
				</View>
			</Modal>
		</View>
	);
}

const styles = StyleSheet.create({
	container: { marginBottom: 16 },
	label: { fontSize: 16, marginBottom: 4 },
	picker: {
		height: 44,
		width: '100%',
		borderWidth: 1,
		borderColor: '#ccc',
		borderRadius: 4,
		justifyContent: 'center',
		paddingHorizontal: 10,
		backgroundColor: '#f9f9f9',
	},
	modalOverlay: {
		flex: 1,
		backgroundColor: 'rgba(0,0,0,0.3)',
		justifyContent: 'center',
		alignItems: 'center',
	},
	modalContent: {
		backgroundColor: '#fff',
		borderRadius: 8,
		padding: 20,
		width: '80%',
		maxHeight: 350,
	},
	option: {
		padding: 12,
		borderBottomWidth: 1,
		borderBottomColor: '#eee',
	},
	selected: {
		color: '#007bff',
		fontWeight: 'bold',
	},
	closeBtn: {
		backgroundColor: '#007bff',
		padding: 10,
		borderRadius: 6,
		alignItems: 'center',
		marginTop: 10,
	},
});
