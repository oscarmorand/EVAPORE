from abc import ABC

class EdgeSplit(ABC):
    def attr_in_data(self, data, attr_name: str) -> bool:
        return hasattr(data, attr_name) and data.__getattr__(attr_name) is not None
    
    def all_attrs_in_data(self, data, attr_names: list) -> bool:
        data_has_all_attr = True
        missing_attrs = []
        for attr_name in attr_names:
            if not self.attr_in_data(data, attr_name):
                data_has_all_attr = False
                missing_attrs.append(attr_name)
        return data_has_all_attr, missing_attrs
    
    def check_basic_data_attrs(self, data) -> None:
        required_attrs = ['x', 'edge_index']
        data_has_all_attr, missing_attrs = self.all_attrs_in_data(data, required_attrs)
        if not data_has_all_attr:
            raise ValueError("Data object is missing required attributes: "
                             f"{', '.join(missing_attrs)}")