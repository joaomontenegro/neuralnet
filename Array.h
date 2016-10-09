#ifndef _ARRAY_H_
#define _ARRAY_H_

#include <stdlib.h>

template<typename T> class Array
{
public:
	Array() : m_size(0), m_elements(0x0) {}
	Array(size_t size) { allocate(size); }

	~Array() { delete[] m_elements; }

	T& operator[](size_t index) { return m_elements[index]; }
	const T& operator[](size_t index) const { return m_elements[index]; }

	T* getRef(size_t index) { return m_elements + index;}

	void allocate(unsigned int newSize)
	{
		if (m_elements != 0x0)
		{
			delete[] m_elements;
		}

		m_size = newSize;
		m_elements = new T[newSize];
	}

	unsigned int size() { return m_size; }

private:
	size_t m_size;
	T* m_elements;
};

#endif
